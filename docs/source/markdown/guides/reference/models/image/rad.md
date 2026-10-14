# RAD

## Description

RAD (Retrieval-based Anomaly Detection) is a training-free framework that stores
anomaly-free features in a multi-layer memory bank and detects anomalies through
multi-level retrieval with position-aware patch matching.

The model uses a DINOv3 ViT-B/16 backbone to extract features from multiple transformer
blocks. At inference, it retrieves the most similar training images via CLS token
similarity, then performs patch-level anomaly scoring using cosine distance with
optional spatial position constraints.

## Usage

```bash
anomalib train \
   --model rad \
   --model.backbone vit_base_patch16_dinov3 \
   --model.layers "[3, 6, 9, 11]" \
   --model.k_image 150 \
   --model.use_positional_bank true \
   --model.pos_radius 1
```

## Key Parameters

| Parameter             | Default                   | Description                                                |
| --------------------- | ------------------------- | ---------------------------------------------------------- |
| `backbone`            | `vit_base_patch16_dinov3` | ViT backbone from timm                                     |
| `layers`              | `[3, 6, 9, 11]`           | Transformer block indices (0-based) for feature extraction |
| `k_image`             | `150`                     | Number of nearest training images for local memory         |
| `use_positional_bank` | `True`                    | Enable position-aware patch matching                       |
| `pos_radius`          | `1`                       | Spatial neighborhood radius for positional matching        |
| `max_ratio`           | `0.01`                    | Fraction of top anomaly pixels for image-level score       |
| `bank_dtype`          | `None`                    | Memory bank storage dtype; `float16` halves its size       |

The defaults follow the paper's MVTec-AD configuration. Other datasets use different values:
`k_image` is 900 for VisA and Real-IAD and 48 for 3D-ADAM, while `pos_radius` is 2 for VisA
and 0 for Real-IAD.

```{note}
Position-aware matching compares patches by their grid position, so the fitting and inference
image sizes must produce the same patch grid. Set `use_positional_bank=False` if they differ.
The memory bank is stored in the model state, so checkpoint size grows with the training set.
Setting `bank_dtype=float16` halves the bank's memory and checkpoint size and speeds up
scoring, at the cost of a small (order `1e-4`) perturbation of the anomaly scores. The CLS
bank that drives image retrieval always stays at full precision, so the set of retrieved
training images is unaffected.
```

```{eval-rst}
.. automodule:: anomalib.models.image.rad.lightning_model
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: anomalib.models.image.rad.torch_model
   :members:
   :show-inheritance:
```
