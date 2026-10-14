# MH-PatchCore

This is the anomalib implementation of
[Mahalanobis PatchCore: Covariance-Aware and Streaming-Compatible Industrial Anomaly Detection](https://arxiv.org/abs/2605.27748),
an arXiv preprint by Niccolò Ferrari, Oligert Osmani, and Evelina Lamma. The
[reference implementation](https://github.com/NickF93/MH-PatchCore) is
available under the MIT license.

Model Type: Segmentation

## Description

MH-PatchCore is a one-class anomaly detector that retains PatchCore's
patch-level nearest-neighbor scoring while learning a covariance-aware feature
space. It projects hierarchical backbone features with PCA, whitens the
projection using a shrinkage covariance estimate, and constructs a bounded
memory bank with deterministic merge-reduce coreset selection.

Unlike standard PatchCore, fitting requires three ordered passes over the
normal training set:

1. Fit the PCA projection to the requested explained-variance ratio.
2. Estimate and factorize the shrunk covariance matrix in PCA space.
3. Whiten the projected patches and build the merge-reduce memory bank.

The implementation uses anomalib's standard model, data, post-processing,
evaluation, visualization, checkpoint, and export interfaces.

## Architecture

![MH-PatchCore architecture](/docs/source/images/mh_patchcore/architecture.png "MH-PatchCore architecture")

The default feature extractor is `wide_resnet50_2.tv2_in1k` from `timm`, using
`layer2` and `layer3`. Its weights correspond to torchvision's
`IMAGENET1K_V2` checkpoint (`wide_resnet50_2-9ba9bcbe.pth`) from
`https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth`, with SHA-256
`9ba9bcbebc349d733a72eb7608143fd754e4689ac8d8ce2916ca0bdff6443950`.
Inputs are resized to 256×256, center-cropped to 224×224, and normalized with
ImageNet statistics.

## Usage

### Python

```python
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import MHPatchcore

datamodule = MVTecAD(
    root="./datasets/MVTecAD",
    category="bottle",
    train_batch_size=32,
    eval_batch_size=32,
)
model = MHPatchcore()
engine = Engine()

engine.fit(model=model, datamodule=datamodule)
results = engine.test(model=model, datamodule=datamodule)
```

### CLI

```bash
anomalib train --model MHPatchcore --data MVTecAD --data.category bottle
```

The repository configuration can also be used directly:

```bash
anomalib train \
    --config examples/configs/model/mh_patchcore.yaml \
    --data MVTecAD \
    --data.category bottle
```

### Constructor arguments

| Argument               | Default                    | Description                                                   |
| ---------------------- | -------------------------- | ------------------------------------------------------------- |
| `backbone`             | `wide_resnet50_2.tv2_in1k` | `timm` feature-extractor name.                                |
| `layers`               | `("layer2", "layer3")`     | Ordered backbone layers used for patch embeddings.            |
| `pre_trained`          | `True`                     | Load pretrained backbone weights.                             |
| `pca_variance_ratio`   | `0.99`                     | Explained variance retained by streaming PCA.                 |
| `covariance_shrinkage` | `0.07`                     | Fixed covariance shrinkage coefficient.                       |
| `memory_bank_size`     | `1000`                     | Maximum number of vectors in the finalized bank.              |
| `local_coreset_size`   | `256`                      | Maximum vectors retained by each merge-reduce block.          |
| `num_neighbors`        | `9`                        | Bank neighbors used to reweight the image score.              |
| `pre_processor`        | `True`                     | Default preprocessor, a custom `PreProcessor`, or disabled.   |
| `post_processor`       | `True`                     | Default postprocessor, a custom `PostProcessor`, or disabled. |
| `evaluator`            | `True`                     | Default evaluator, a custom `Evaluator`, or disabled.         |
| `visualizer`           | `True`                     | Default visualizer, a custom `Visualizer`, or disabled.       |

During inference, the Torch model returns an image-level `pred_score` and a
pixel-level `anomaly_map`. Standard anomalib post-processing adds thresholded
labels and masks when enabled.

## Lifecycle and limitations

- Fitting uses exactly three epochs, one for each statistical pass. Validation
  runs only after the memory bank has been finalized.
- Fitting is fixed to one device because distributed fitting would partition
  the streaming statistics.
- Checkpoints saved at an epoch boundary or after fitting can be restored.
  Checkpoints captured partway through a fitting pass are rejected because the
  transient accumulators are intentionally not serialized.
- Torch, ONNX, and OpenVINO export are supported after the model has been fully
  fitted.
- The canonical 224×224 crop and default backbone/layers are the validated
  configuration. Other `timm` backbones require compatible layer names and have
  not been benchmarked here.

## Benchmarks

All tables below use the 15 MVTec AD categories, but they describe distinct
experiments and should not be compared as if their runtimes shared a protocol.

### Published historical results

These values are reported by the preprint for its canonical CPU experiment.
They were produced with the paper's software and hardware environment; the
timing and memory columns are not directly comparable with the measurements
below.

| Image AUROC | Pixel AUROC | Mean fit (s) | Inference (ms/image) | Mean RAM (GB) | Peak RAM (GB) |
| ----------: | ----------: | -----------: | -------------------: | ------------: | ------------: |
|       0.989 |       0.978 |       57.913 |               42.610 |         2.430 |         2.775 |

### Native anomalib results

The native benchmark used anomalib's public `MHPatchcore` and `MVTecAD` APIs,
seed 42, shuffled training, batch size 32, eight data workers, and the canonical
preprocessing and weights. It ran on CPU on an Intel Core i7-10750H with
anomalib revision `9cb05fb7`, Python 3.12.8, PyTorch 2.14.0, torchvision 0.29.0,
and `timm` 1.0.29.

| Category   | Image AUROC |  Image F1 | Pixel AUROC |  Pixel F1 |
| ---------- | ----------: | --------: | ----------: | --------: |
| Bottle     |       1.000 |     0.992 |       0.985 |     0.781 |
| Cable      |       0.969 |     0.920 |       0.964 |     0.631 |
| Capsule    |       0.988 |     0.977 |       0.986 |     0.525 |
| Carpet     |       0.990 |     0.971 |       0.988 |     0.605 |
| Grid       |       0.980 |     0.973 |       0.983 |     0.436 |
| Hazelnut   |       1.000 |     0.993 |       0.984 |     0.615 |
| Leather    |       1.000 |     0.995 |       0.992 |     0.488 |
| Metal Nut  |       1.000 |     0.995 |       0.988 |     0.883 |
| Pill       |       0.972 |     0.968 |       0.984 |     0.744 |
| Screw      |       0.940 |     0.928 |       0.980 |     0.441 |
| Tile       |       0.992 |     0.982 |       0.962 |     0.661 |
| Toothbrush |       1.000 |     0.983 |       0.986 |     0.584 |
| Transistor |       1.000 |     0.987 |       0.935 |     0.572 |
| Wood       |       0.996 |     0.975 |       0.953 |     0.542 |
| Zipper     |       0.998 |     0.983 |       0.987 |     0.662 |
| **Mean**   |   **0.988** | **0.975** |   **0.977** | **0.611** |

Mean fitting time was 626.12 seconds per category, image-weighted inference
time was 0.446 seconds per image, and the mean/maximum process resident memory
was 4541/4585 MiB. Runtime and memory are descriptive measurements rather than
accuracy criteria.

### Same-machine ordered comparison

For the numerical comparison, both the reference implementation at
`fecde8b1` and the anomalib implementation at `9cb05fb7` used the same sample
order, seed 42, batch size 16, no data workers, CPU float32 inference, and one
numerical thread. The reference environment used Python 3.11.11, PyTorch
2.10.0, torchvision 0.25.0, and `timm` 1.0.24; the anomalib environment used
the versions listed above.

| Implementation | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 | Mean fit (s) | Inference (s/image) | Mean/peak RSS (MiB) |
| -------------- | ----------: | -------: | ----------: | -------: | -----------: | ------------------: | ------------------: |
| Reference      |    0.987788 | 0.978983 |    0.977663 | 0.610917 |       871.38 |               0.911 |           2675/2727 |
| anomalib       |    0.987788 | 0.978983 |    0.977663 | 0.610917 |       988.61 |               1.056 |           5438/5518 |

Across categories, the maximum absolute differences were 0 for image AUROC and
image F1, `3.95e-08` for pixel AUROC, and `5.23e-06` for pixel F1.

### Sample results

![Bottle result](/docs/source/images/mh_patchcore/results/0.png "Bottle result")

![Leather result](/docs/source/images/mh_patchcore/results/1.png "Leather result")

![Metal nut result](/docs/source/images/mh_patchcore/results/2.png "Metal nut result")

## License and attribution

The anomalib integration is distributed under Apache-2.0. Material derived from
the MIT-licensed reference implementation retains the notice in this package's
`LICENSE` file and in anomalib's third-party inventory. The reference code is
copyright Niccolò Ferrari and Oligert Osmani.

## Reference

```bibtex
@article{ferrari2026mahalanobis,
  title={Mahalanobis PatchCore: Covariance-Aware and Streaming-Compatible Industrial Anomaly Detection},
  author={Ferrari, Niccolò and Osmani, Oligert and Lamma, Evelina},
  journal={arXiv preprint arXiv:2605.27748},
  year={2026}
}
```
