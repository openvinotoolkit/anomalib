# FoundAD

## Description

FoundAD trains a lightweight manifold projector on top of a frozen DINOv2/DINOv3
encoder, using CutPaste-synthesized anomalies for supervision. At inference, the
MSE between encoder and projector features flags off-manifold (anomalous) patches.
Effective in few-shot settings with only 1-4 normal training images. See the
[model README](https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/models/image/foundad/README.md)
for the full architecture description and benchmark results.

## Usage

```bash
anomalib train --model FoundAD --data MVTecAD --data.category bottle
```

```{eval-rst}
.. automodule:: anomalib.models.image.foundad.lightning_model
   :members: FoundAD
   :show-inheritance:
```

```{eval-rst}
.. automodule:: anomalib.models.image.foundad.torch_model
   :members: FoundADModel
   :show-inheritance:
```
