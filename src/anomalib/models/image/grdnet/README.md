# GRD-Net

GRD-Net combines a generative-reconstructive network with a discriminative network for one-class image anomaly detection and localization. This port follows [GRD-Net: Generative-Reconstructive-Discriminative Anomaly Detection with Region of Interest Attention Module](https://doi.org/10.1155/2023/7773481), by Niccolò Ferrari, Michele Fraccaroli, and Evelina Lamma (2023).

The [public PyTorch implementation](https://github.com/NickF93/GRD-Net) is MIT licensed. This implementation uses the paper's 32-channel spatial latent representation, independent encoder projections, and anomalib's existing DRÆM segmentator. Where the paper gives conflicting ROI notation, it follows Figure 3: the ROI masks the synthetic training target, not the prediction. The original experimental image/patch geometry could not be established, so the geometry below is an explicit implementation choice.

## Architecture

![GRD-Net training and inference](/docs/source/images/grdnet/architecture.svg)

Images are resized to 256 × 256 without ImageNet normalization. Internal 128 × 128 tiles with stride 64 produce nine overlapping tiles per image. Training rotates each tile and its optional ROI together, then alpha-blends a texture inside a Perlin mask. It does not add flips or global Gaussian noise.

The generator has two independently parameterized residual encoders, a mirrored decoder, and independent 32 × 8 × 8 latent projections. It reconstructs the clean tile from the corrupted tile. An adversarial discriminator provides feature matching; the existing DRÆM discriminative subnetwork predicts two-class segmentation logits from the corrupted tile and its detached reconstruction.

Training uses three separate Adam steps: discriminator, generator, then segmentator. The generator combines feature MSE, contextual L1 plus one minus SSIM, and latent L1, weighted 1, 50, and 1. Discriminator parameters and batch-normalization statistics are frozen during the generator step. The segmentation loss is focal loss with gamma 2 and no alpha class balancing. Only the generator learning rate is reduced on a plateau in mean training contextual loss, by a factor of exp(-0.1).

At inference, clean tiles and their reconstructions enter the segmentator. Anomalous-class probabilities are averaged across overlaps and smoothed with a 21 × 21 average filter. The maximum smoothed value is the image anomaly score. The low-level model returns an `InferenceBatch` with scores shaped `[B]` and maps shaped `[B, 1, 256, 256]`; the ordinary postprocessor can additionally normalize scores and derive labels/masks.

## Usage

### Python

```python
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import GRDNet

datamodule = MVTecAD(
    root="datasets/mvtec_anomaly_detection",
    category="hazelnut",
    train_batch_size=4,
    eval_batch_size=1,
)
model = GRDNet()
engine = Engine(max_epochs=200)
engine.fit(model=model, datamodule=datamodule)
engine.test(model=model, datamodule=datamodule)
```

### CLI

```bash
anomalib train --model GRDNet --data MVTecAD \
  --data.root datasets/mvtec_anomaly_detection --data.category hazelnut \
  --data.train_batch_size 4 --data.eval_batch_size 1 --trainer.max_epochs 200
```

### YAML

Save this as `grdnet.yaml` and run `anomalib train --config grdnet.yaml`:

```yaml
model:
  class_path: anomalib.models.GRDNet
  init_args:
    texture_source: random
data:
  class_path: anomalib.data.MVTecAD
  init_args:
    root: datasets/mvtec_anomaly_detection
    category: hazelnut
    train_batch_size: 4
    eval_batch_size: 1
trainer:
  max_epochs: 200
```

These are ordinary anomalib workflows, not the test-oracle checkpoint-selection procedure used in the measurements below. Training duration and checkpoint callbacks remain under Engine/Trainer control.

### Model options

| Parameter            | Default    | Meaning                                                                                                 |
| -------------------- | ---------- | ------------------------------------------------------------------------------------------------------- |
| `texture_source`     | `"random"` | Independent uniform RGB texture; `"image"` instead circularly shifts the same tile by a nonzero offset. |
| `perlin_probability` | `0.75`     | Probability of synthesizing an anomaly in each tile.                                                    |
| `adversarial_weight` | `1.0`      | Generator feature-matching weight.                                                                      |
| `contextual_weight`  | `50.0`     | Generator L1 plus one-minus-SSIM weight.                                                                |
| `encoder_weight`     | `1.0`      | Latent-consistency weight.                                                                              |
| `learning_rate`      | `1e-4`     | Initial learning rate of all three Adam optimizers.                                                     |

Both texture modes use the same Perlin mask and blend factor sampled uniformly from 0.5 to 1.0. No external texture dataset is required. The architecture, internal tiling, and smoothing are fixed. The usual `pre_processor`, `post_processor`, `evaluator`, and `visualizer` arguments accept instances or booleans and default to `True`. Custom preprocessing must not include `Normalize`; images must remain in `[0, 1]`.

### Optional ROI supervision

ROI masks describe where synthetic anomalies should be treated as relevant during training. They are **not** anomaly ground truth and must not be supplied as `gt_mask`. The segmentator target is the synthetic mask intersected with the ROI. Inference neither requires ROI masks nor multiplies them into predictions.

```python
from anomalib.data import GRDNetFolder
from anomalib.engine import Engine
from anomalib.models import GRDNet

datamodule = GRDNetFolder(
    name="widgets",
    root="datasets/widgets",
    normal_dir="train/good",
    normal_test_dir="test/good",
    abnormal_dir="test/bad",
    mask_dir="ground_truth/bad",
    roi_dir="roi",
    train_batch_size=4,
    eval_batch_size=1,
)
engine = Engine(max_epochs=200)
engine.fit(model=GRDNet(), datamodule=datamodule)
```

ROI paths mirror image paths relative to `root`. For `train/good/000.jpg`, the dataset first looks for `roi/train/good/000.jpg`, then `roi/train/good/000.png`. An absolute ROI directory is also supported; resolved masks must remain inside it. A missing ROI uses a full-image mask. If a configured ROI directory exists but some files are absent, the dataset logs a summarized warning. An invalid configured directory fails rather than silently disabling ROI supervision.

`GRDNetItem` and `GRDNetBatch` extend the standard Torch image dataclasses with `roi_mask` and `roi_mask_path`. Item masks are `[H, W]`; batch masks are `[B, H, W]`. Dataset transforms and default preprocessing keep the image, anomaly ground truth, and ROI aligned. NumPy conversion deliberately returns ordinary image dataclasses without the training-only ROI fields. Standard datamodules such as `MVTecAD` need no ROI configuration.

## Checkpoints and export

Standard Lightning checkpoints contain the three subnetworks, optimizers, and generator scheduler. Pass `ckpt_path` to the usual Engine methods to restore or resume. The contextual epoch mean is reset each epoch rather than persisted as training history.

Torch, ONNX, and OpenVINO export use images only, at fixed 256 × 256 input size. The inference graph contains reconstruction, segmentation, overlap averaging, smoothing, and scoring—not ROI loading or synthetic anomaly generation:

```python
from anomalib.deploy import ExportType

engine.export(
    model=model,
    export_type=ExportType.ONNX,
    ckpt_path="results/GRDNet/checkpoint.ckpt",
    input_size=(256, 256),
)
```

## Selected-category MVTec AD measurements

The following measurements cover Hazelnut, Metal Nut, and Pill, with seed 1337, random textures, no ROI, float32, training batch size 4, and evaluation batch size 1. Twenty percent of normal training images were held out for synthetic-SSIM diagnostics. The complete test split was evaluated after every epoch, and image and raw-pixel AUROC maxima were selected **independently using test labels**. Later ties replaced earlier selections. This is test-oracle selection, not an unbiased estimate of generalization.

Image scores use the smoothed map maximum. The raw-pixel selection uses the unsmoothed map; the table also shows the public smoothed-map AUROC at that pixel-selected checkpoint. Image and pixel maxima generally belong to different checkpoints and must not be interpreted as the performance of one fitted model.

The compact [measured artifact](/results/GRDNet/selected_categories.json) records checkpoint hashes, metrics, curves, protocol, and sample provenance. These measurements do not establish complete reproduction of the paper.

### Paper-reported results

| Category  | Image AUROC | Pixel AUROC |
| --------- | ----------- | ----------- |
| Hazelnut  | 1.000       | 0.974       |
| Metal Nut | 1.000       | 0.962       |
| Pill      | 0.985       | 0.958       |

Source: Table 5 of the paper, 200 epochs. The original input geometry and complete experimental protocol were not recovered; these are historical values, not measurements from this port.

### Independent selections through 200 epochs

| Category  | Image AUROC | Image epoch | Raw pixel AUROC | Pixel epoch | Smoothed pixel AUROC at pixel checkpoint |
| --------- | ----------- | ----------- | --------------- | ----------- | ---------------------------------------- |
| Hazelnut  | 0.9879      | 36          | 0.8276          | 133         | 0.9312                                   |
| Metal Nut | 0.9638      | 168         | 0.7989          | 190         | 0.8894                                   |
| Pill      | 0.9435      | 119         | 0.8926          | 169         | 0.9608                                   |

### Independent selections through 300 epochs

The additional 100 epochs are a separately reported observation window, not the paper's 200-epoch protocol.

| Category  | Image AUROC | Image epoch | Raw pixel AUROC | Pixel epoch | Smoothed pixel AUROC at pixel checkpoint |
| --------- | ----------- | ----------- | --------------- | ----------- | ---------------------------------------- |
| Hazelnut  | 0.9961      | 278         | 0.8276          | 133         | 0.9312                                   |
| Metal Nut | 0.9638      | 168         | 0.8412          | 218         | 0.8902                                   |
| Pill      | 0.9534      | 257         | 0.9158          | 221         | 0.9756                                   |

### Hazelnut geometry sensitivity

A single whole-image 256 × 256 comparison used the same seed and selection procedure. It is an external diagnostic profile, not a public model option or evidence of the paper's original geometry.

| Geometry              | Epoch window | Image AUROC | Image epoch | Raw pixel AUROC | Pixel epoch | Smoothed pixel AUROC at pixel checkpoint |
| --------------------- | ------------ | ----------- | ----------- | --------------- | ----------- | ---------------------------------------- |
| Tiles 128 / stride 64 | 1–200        | 0.9879      | 36          | 0.8276          | 133         | 0.9312                                   |
| Whole image 256       | 1–200        | 0.9939      | 127         | 0.9070          | 56          | 0.9528                                   |
| Tiles 128 / stride 64 | 1–300        | 0.9961      | 278         | 0.8276          | 133         | 0.9312                                   |
| Whole image 256       | 1–300        | 0.9939      | 127         | 0.9070          | 56          | 0.9528                                   |

### Runtime and resources

Measured on an NVIDIA RTX 3080 with 10 GB VRAM, Linux x86-64, Python 3.12.8, Torch 2.14.0, torchvision 0.29.0, and Lightning 2.6.6, at implementation revision `08d11cee`. Fit time includes the full 300 epochs and per-epoch test evaluation. Prediction time is total `Engine.predict` wall time divided by test-image count, including data loading and preprocessing, without visualization; it is not isolated kernel latency.

| Category  | Fit time (s) | Prediction (ms/image) | Peak RAM (MiB) | Peak allocated VRAM (MiB) | Peak reserved VRAM (MiB) |
| --------- | ------------ | --------------------- | -------------- | ------------------------- | ------------------------ |
| Hazelnut  | 24977.34     | 33.38                 | 3914.16        | 7872.90                   | 8868.00                  |
| Metal Nut | 15277.41     | 33.47                 | 3897.00        | 7872.90                   | 8868.00                  |
| Pill      | 19003.89     | 32.91                 | 3943.87        | 7872.90                   | 8868.00                  |

RAM is the process peak from `resource.getrusage`; VRAM uses Torch's allocator peak counters. Batch size 4 processes 36 internal tiles and fits this GPU. Reduce the training batch for smaller GPUs; the datamodule defaults are not a memory-capacity recommendation.

## Sample results

The panels use each category's raw-pixel-selected checkpoint through epoch 200 and display its **smoothed** anomaly map. Heatmaps are min-max scaled per image for visualization only; panel captions state the original probability range. They are qualitative examples, not additional benchmark results.

![Hazelnut prediction](/docs/source/images/grdnet/results/0.png)
![Metal Nut prediction](/docs/source/images/grdnet/results/1.png)
![Pill prediction](/docs/source/images/grdnet/results/2.png)

Dataset images and annotations are from MVTec Software GmbH. These dataset-derived panels retain [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); see the [asset notice](/docs/source/images/grdnet/results/LICENSE). They are not Apache-2.0 assets.

## Reference

Ferrari, N., Fraccaroli, M., and Lamma, E. (2023). _GRD-Net: Generative-Reconstructive-Discriminative Anomaly Detection with Region of Interest Attention Module_. International Journal of Intelligent Systems, Article 7773481. [doi:10.1155/2023/7773481](https://doi.org/10.1155/2023/7773481).

The model's colocated [LICENSE](LICENSE) preserves upstream MIT attribution. The DRÆM segmentator is imported from anomalib rather than copied. No proprietary datasets or historical TensorFlow code are included.
