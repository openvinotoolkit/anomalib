# PatchCore

## Architecture

```{eval-rst}
.. image:: ../../../../../images/patchcore/architecture.jpg
    :alt: PatchCore Architecture
```

## Using Different Backbones

PatchCore extracts intermediate feature maps from a backbone network. When changing the backbone, the selected `layers` must match valid layer names in that backbone.

Example CLI usage:

```bash
anomalib train \
   --model patchcore \
   --model.backbone wide_resnet50_2 \
   --model.layers layer2 layer3 \
   --model.pre_trained true
```

Common examples:

| Backbone                | Example layers          |
| ----------------------- | ----------------------- |
| `resnet18`              | `layer2 layer3`         |
| `resnet50`              | `layer2 layer3`         |
| `wide_resnet50_2`       | `layer2 layer3`         |
| `mobilenetv3_large_100` | `blocks.4.1 blocks.6.0` |

The correct layer names depend on the selected backbone architecture. PatchCore uses
`timm` feature extraction, so valid layer names can be inspected with:

```python
import timm

model = timm.create_model(
    "mobilenetv3_large_100",
    pretrained=False,
    features_only=True,
    exportable=True,
)

print([info["module"] for info in model.feature_info.info])
```

Unknown layer names may be ignored with a warning. If no valid layers remain after
filtering, PatchCore will later fail because no features are extracted.

## Memory

During training, patch embeddings are copied into a single preallocated staging
tensor as each batch arrives (see
`anomalib.models.image.patchcore.embedding_store`). The staging capacity is
reserved exactly from the training dataloader length after the first batch, so
no second full-size copy of the embedding matrix is materialized at fit time.
If the training loop cannot provide a usable dataloader length, the store
falls back to the historical per-batch list, which is always correct.

```{eval-rst}
.. automodule:: anomalib.models.image.patchcore.lightning_model
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: anomalib.models.image.patchcore.torch_model
   :members:
   :show-inheritance:
```
