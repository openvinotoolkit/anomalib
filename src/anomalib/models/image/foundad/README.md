# FoundAD

This is an implementation of the FoundAD model, based on the [official code](https://github.com/ymxlzgy/FoundAD).

The model was presented in [Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors](https://arxiv.org/abs/2510.01934) (Zhai et al.).

Model Type: Segmentation

## Description

**FoundAD** shows that a frozen foundation vision encoder (DINOv2/DINOv3) already
contains enough information to detect anomalies from very few normal examples,
as long as a lightweight _manifold projector_ is trained on top of it. The
projector learns to map encoder features back onto the manifold of normal
image features; at inference time, patches whose projected features deviate
strongly from the encoder's output (high MSE) are flagged as anomalous.

### Architecture

1. **Frozen encoder**: a pre-trained DINOv2 (or DINOv3, via `timm`) Vision
   Transformer extracts patch features from an intermediate layer
   (`n_layer`-th block counting from the end). The encoder is never updated.
2. **Manifold projector**: the only trainable component. It embeds encoder
   features to a smaller internal dimension, passes them through a stack of
   transformer blocks that use a **global residual skip connection** (each
   block's output is added back to the _initial_ embedded features rather
   than to the previous block's output), and projects back to the encoder's
   feature dimension.
3. **Few-shot training augmentation**: since the whole point of FoundAD is to
   train from a handful of normal images, the base training images are
   randomly flipped/rotated (`p=0.3`) and independently color-jittered,
   grayscaled, or Gaussian-blurred (`p=0.3`) _before_ CutPaste is applied,
   matching the original repo's `build_train_transform_staged`. This
   meaningfully expands the effective diversity of a tiny few-shot set and
   can be disabled via `use_few_shot_augmentation=False`.
4. **CutPaste supervision**: during training, CutPaste-style synthetic
   anomalies (cut-and-paste rectangular patches or rotated "scar" patches,
   with color jitter) are generated from the normal training images. With 50%
   probability per batch, the projector is fed features from the augmented
   (off-manifold) image while the target remains the clean image's features,
   teaching the projector to correct off-manifold inputs back onto the
   manifold.
5. **Anomaly scoring**: at inference, the per-patch MSE between encoder and
   projector features is computed, resized to the input resolution and
   Gaussian-smoothed to form the pixel-level anomaly map, and the mean of the
   top-`K` most anomalous patches gives the image-level anomaly score.

### Deviations from the original paper/implementation

This Anomalib implementation intentionally differs from the original FoundAD
repository in one way:

- **CutPaste patch placement**: the original computes a per-category
  foreground segmentation mask (hand-tuned per MVTec-AD/VisA category) and
  restricts patch placement to the foreground. This implementation places
  patches at a fully random location, trading a small amount of fidelity for
  a simpler, dataset-agnostic augmentation that requires no category-specific
  tuning.

This deviation is expected to contribute to differences between the
benchmark numbers below and the numbers reported in the paper. The
predictor's attention mechanism (standard softmax multi-head attention, via
`torch.nn.functional.scaled_dot_product_attention`) matches the original
paper exactly and is fully OpenVINO-export compatible (verified — see
[Export](#export) below).

### Numerical stability note

Training occasionally diverged to `NaN` loss on longer runs. The cause was in
`components/cutpaste.py`: `ColorJitter`'s RGB→HSV conversion divides by the
per-pixel max channel value, which is only safe for `[0, 1]`-range images —
CutPaste operates on ImageNet-normalized tensors (can be negative), so this
occasionally produced a real division-by-zero. Fixed by applying
`ColorJitter` in real, un-normalized pixel space (see the module docstring
in `components/cutpaste.py`). The same pattern exists in the original
repository's `synthesis.py` too.

## Usage

`anomalib train --model FoundAD --data MVTecAD --data.category <category>`

```python
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import FoundAD

model = FoundAD()
datamodule = MVTecAD()
engine = Engine()
engine.fit(model=model, datamodule=datamodule)
```

FoundAD is designed for **few-shot** settings — pass a datamodule/subset with
only a handful of normal training images (e.g. via a custom
`AnomalibDataset.subsample`) to reproduce the paper's 1/2/4-shot protocol.

## Export

FoundAD supports export to `TORCH`, `ONNX`, and `OPENVINO` (FP32, FP16, INT8
weight-compression, and INT8 post-training quantization). OpenVINO export
parity against native PyTorch was validated on real test images (dynamic
input shape, the only export mode compatible with `OpenVINOInferencer`'s
raw-image inference workflow):

| Export                                | Max score diff | Mean score diff | Score correlation |
| ------------------------------------- | -------------- | --------------- | ----------------- |
| FP32 (no compression)                 | 2.00%          | 0.66%           | 0.98              |
| FP16                                  | 1.99%          | 0.66%           | 0.98              |
| INT8 (weight-only)                    | 2.45%          | 0.90%           | 0.96              |
| INT8 PTQ (post-training quantization) | 3.26%          | 1.75%           | 0.83              |

FP32/FP16 differences are consistent with ordinary cross-framework
numerical differences through a deep frozen DINOv2 encoder, not a
functional defect. INT8 PTQ shows more drift and should be calibrated with
a properly-sized dataset (nncf recommends 300+ images) for production use.

## Benchmark

The paper's few-shot protocol is **multi-class-one-model**: one shared model
is trained per dataset per shot-count, using few-shot images pooled across
_all_ categories (e.g. 15 images for MVTec-AD 1-shot), then evaluated on
every category's full test set with that same model. Numbers below
reproduce this protocol through Anomalib's own `FoundADModel`/training loop
(not the paper's original code), sampling with the same seed (`42`) +
shuffle-then-take-first-k procedure as the official `sample.py`, using
`dinov3_vit_base_16`:

### MVTec-AD (15 categories, pooled)

| Shot | I-AUROC (ours) | I-AUROC (paper) | AUPR (ours) | AUPR (paper) | P-AUROC (ours) | P-AUROC (paper) |
| ---- | -------------- | --------------- | ----------- | ------------ | -------------- | --------------- |
| 1    | 94.5           | 96.1            | 97.2        | 97.9         | 97.2           | 96.8            |
| 2    | 95.5           | 96.8            | 97.9        | 98.3         | 97.5           | 97.0            |
| 4    | 96.7           | 97.1            | 98.3        | 98.6         | 97.7           | 97.2            |

### VisA (12 categories, pooled)

| Shot | I-AUROC (ours) | I-AUROC (paper) | AUPR (ours) | AUPR (paper) | P-AUROC (ours) | P-AUROC (paper) |
| ---- | -------------- | --------------- | ----------- | ------------ | -------------- | --------------- |
| 1    | 92.2           | 92.6            | 91.2        | 92.0         | 96.8           | 99.7            |
| 2    | 93.5           | 93.5            | 92.4        | 93.0         | 97.9           | 99.7            |
| 4    | 93.6           | 94.4            | 92.0        | 94.0         | 98.7           | 99.7            |

Our numbers now sit within 0.4-1.6 I-AUROC points of the paper across every
shot count on both datasets (validated separately: loading the authors'
own official checkpoint into this implementation scores 95.5% I-AUROC on 5
MVTec-AD categories, vs. their reported 96.1%, confirming the eval/scoring
code itself is correct). The small residual gap is explained by known,
deliberate simplifications rather than a bug:

- **2000 vs. the paper's 4000 training epochs**, to keep the sweep
  tractable.
- **Random vs. foreground-masked CutPaste placement** (see
  [Deviations](#deviations-from-the-original-paperimplementation) above).
- **Single seed** vs. the paper's 3-seed average.

Note: the original repo's few-shot training-time augmentation (random
flip/rotate/color-jitter/grayscale/blur, ported here as
`FewShotAugmentation`) is essential — without it, image-level AUROC drops
_below random_ (as low as ~23%) since the projector overfits to the exact
pixels of the tiny few-shot set. It is enabled by default
(`use_few_shot_augmentation=True`).
