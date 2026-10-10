# GRD-Net Folder Datamodule

`GRDNetFolder` extends `Folder` with optional ROI supervision for GRD-Net. ROI masks identify where synthetic anomalies matter during training. They remain separate from anomaly ground-truth masks and are not required for inference.

## Folder layout

Image paths must resolve inside `root`. ROI paths mirror those image paths relative to `root`:

```text
widgets/
├── train/good/000.jpg
├── test/good/000.jpg
├── test/bad/000.jpg
├── ground_truth/bad/000.png
└── roi/train/good/000.png
```

For `train/good/000.jpg`, the ROI lookup prefers `roi/train/good/000.jpg` and falls back to `roi/train/good/000.png`. An absolute `roi_dir` is supported, but every resolved ROI file must stay within that directory, including symlink targets.

No `roi_dir` means full-image ROI masks without warnings. Missing individual files in an existing ROI directory produce a summarized warning and full-image fallbacks. An invalid configured directory raises an error. Fallback masks start at native image size; image, ground truth, and ROI then receive joint torchvision v2 transforms. Masks use nearest-neighbor interpolation.

## Python

```python
from anomalib.data import GRDNetFolder
from anomalib.engine import Engine
from anomalib.models import GRDNet

data = GRDNetFolder(
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
engine.fit(model=GRDNet(), datamodule=data)
```

## CLI

```bash
anomalib train --model GRDNet --data GRDNetFolder \
  --data.name widgets --data.root datasets/widgets \
  --data.normal_dir train/good --data.normal_test_dir test/good \
  --data.abnormal_dir test/bad --data.mask_dir ground_truth/bad \
  --data.roi_dir roi --data.train_batch_size 4 --data.eval_batch_size 1 \
  --trainer.max_epochs 200
```

## YAML

Save as `grdnet-roi.yaml` and run `anomalib train --config grdnet-roi.yaml`:

```yaml
model:
  class_path: anomalib.models.GRDNet
data:
  class_path: anomalib.data.GRDNetFolder
  init_args:
    name: widgets
    root: datasets/widgets
    normal_dir: train/good
    normal_test_dir: test/good
    abnormal_dir: test/bad
    mask_dir: ground_truth/bad
    roi_dir: roi
    train_batch_size: 4
    eval_batch_size: 1
trainer:
  max_epochs: 200
```

Batch size 4 is a memory-conscious example, not a change to the datamodule's defaults. The model processes nine internal tiles per image.

## ROI dataclasses

`GRDNetItem` and `GRDNetBatch` subclass the ordinary Torch image dataclasses. Their additional fields are:

| Field           | Item                          | Batch                            |
| --------------- | ----------------------------- | -------------------------------- |
| `roi_mask`      | Optional boolean `Mask[H, W]` | Optional boolean `Mask[B, H, W]` |
| `roi_mask_path` | Optional string               | Optional list of strings         |

The folder dataset materializes a full-image mask when needed and uses an empty path string for fallback items, keeping collation stable. `gt_mask` and `mask_path` retain their ordinary anomaly-ground-truth meanings.

NumPy conversion intentionally returns the standard `NumpyImageItem` or `NumpyImageBatch` without training-only ROI fields. Existing visualization therefore uses the shared image/prediction fields. Ordinary `ImageBatch` inputs are also supported: GRD-Net supplies an all-ones training ROI.

## API

```{eval-rst}
.. automodule:: anomalib.data.datamodules.image.grdnet
   :members: GRDNetFolder
   :show-inheritance:
```

```{eval-rst}
.. automodule:: anomalib.data.datasets.image.grdnet
   :members: GRDNetFolderDataset
   :show-inheritance:
```

```{eval-rst}
.. automodule:: anomalib.data.dataclasses.torch.grdnet
   :members: GRDNetItem, GRDNetBatch
   :show-inheritance:
```
