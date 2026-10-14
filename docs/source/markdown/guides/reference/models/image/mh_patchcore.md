# MH-PatchCore

MH-PatchCore is a one-class image anomaly detector based on the
[Mahalanobis PatchCore](https://arxiv.org/abs/2605.27748) arXiv preprint by
Niccolò Ferrari, Oligert Osmani, and Evelina Lamma. It retains PatchCore's
patch-level nearest-neighbor scoring and adds PCA projection, shrinkage
covariance whitening, and a bounded merge-reduce memory bank. The
[reference implementation](https://github.com/NickF93/MH-PatchCore) is
MIT-licensed.

## Architecture

```{eval-rst}
.. image:: ../../../../../images/mh_patchcore/architecture.png
    :alt: MH-PatchCore architecture
```

Fitting makes three ordered passes over the normal training images:

1. PCA estimation.
2. Covariance estimation and Cholesky factorization in PCA space.
3. Whitened memory-bank construction.

At inference time, the same feature path projects and whitens each patch before
nearest-neighbor scoring. The patch grid produces the anomaly map, while the
most anomalous patch and its local memory-bank support produce the image score.

## Usage

```python
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import MHPatchcore

datamodule = MVTecAD(root="./datasets/MVTecAD", category="bottle")
model = MHPatchcore()
engine = Engine()

engine.fit(model=model, datamodule=datamodule)
engine.test(model=model, datamodule=datamodule)
```

The model is also available through the CLI:

```bash
anomalib train --model MHPatchcore --data MVTecAD --data.category bottle
```

To use the repository configuration:

```bash
anomalib train \
    --config examples/configs/model/mh_patchcore.yaml \
    --data MVTecAD \
    --data.category bottle
```

## Default configuration

| Argument               | Default                    |
| ---------------------- | -------------------------- |
| `backbone`             | `wide_resnet50_2.tv2_in1k` |
| `layers`               | `("layer2", "layer3")`     |
| `pre_trained`          | `True`                     |
| `pca_variance_ratio`   | `0.99`                     |
| `covariance_shrinkage` | `0.07`                     |
| `memory_bank_size`     | `1000`                     |
| `local_coreset_size`   | `256`                      |
| `num_neighbors`        | `9`                        |
| `pre_processor`        | `True`                     |
| `post_processor`       | `True`                     |
| `evaluator`            | `True`                     |
| `visualizer`           | `True`                     |

The default backbone uses the `IMAGENET1K_V2` weights stored as
`wide_resnet50_2-9ba9bcbe.pth` (SHA-256
`9ba9bcbebc349d733a72eb7608143fd754e4689ac8d8ce2916ca0bdff6443950`).
Canonical preprocessing resizes images to 256×256, takes a 224×224 center crop,
and applies ImageNet normalization. The raw model returns `pred_score` and
`anomaly_map`; anomalib post-processing adds thresholded labels and masks.

## Lifecycle and limitations

- Fitting requires exactly three epochs and one device. Validation begins only
  after the third pass finalizes the memory bank.
- Epoch-boundary and fully fitted checkpoints can be restored. Checkpoints
  captured partway through a fitting pass are rejected because transient
  accumulators are not serialized.
- Torch, ONNX, and OpenVINO export are supported after fitting.
- The 224×224 crop with the default backbone and layers is the validated
  configuration. Alternative `timm` backbones need compatible layer names and
  are not covered by the reported benchmarks.

Detailed MVTec AD results and their separate published, native anomalib, and
same-machine ordered protocols are available in the model package README.

## Sample results

```{eval-rst}
.. image:: ../../../../../images/mh_patchcore/results/0.png
    :alt: MH-PatchCore bottle result

.. image:: ../../../../../images/mh_patchcore/results/1.png
    :alt: MH-PatchCore leather result

.. image:: ../../../../../images/mh_patchcore/results/2.png
    :alt: MH-PatchCore metal nut result
```

## API reference

```{eval-rst}
.. automodule:: anomalib.models.image.mh_patchcore.lightning_model
   :members: MHPatchcore
   :show-inheritance:
```

```{eval-rst}
.. automodule:: anomalib.models.image.mh_patchcore.torch_model
   :members: MHPatchcoreModel
   :show-inheritance:
```
