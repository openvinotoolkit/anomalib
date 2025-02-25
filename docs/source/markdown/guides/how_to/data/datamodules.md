```{eval-rst}
:orphan:
```

# Datamodules

This guide explains how Lightning DataModules work in Anomalib and how they integrate with {doc}`datasets <./datasets>` and {doc}`dataclasses <./dataclasses>`.

## Overview

DataModules encapsulate all the steps needed to process data:

- Download/prepare the data
- Set up train/val/test datasets
- Apply transforms
- Create data loaders

## Basic Structure

A typical Anomalib DataModule follows this structure:

```python
from lightning.pytorch import LightningDataModule
from anomalib.data.datasets.base.image import AnomalibDataset
from torch.utils.data import DataLoader

class AnomalibDataModule(LightningDataModule):
    def __init__(
        self,
        root: str = "./datasets",
        category: str = "bottle",
        image_size: tuple[int, int] = (256, 256),
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        transform = None,
    ):
        super().__init__()
        self.root = root
        self.category = category
        self.image_size = image_size
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.transform = transform
```

## Integration with Datasets

DataModules create and manage dataset instances:

```python
def setup(self, stage: str | None = None):
    """Set up train, validation and test datasets."""
    if stage == "fit" or stage is None:
        self.train_dataset = AnomalibDataset(
            root=self.root,
            category=self.category,
            transform=self.transform,
            split="train"
        )

        self.val_dataset = AnomalibDataset(
            root=self.root,
            category=self.category,
            transform=self.transform,
            split="val"
        )

    if stage == "test" or stage is None:
        self.test_dataset = AnomalibDataset(
            root=self.root,
            category=self.category,
            transform=self.transform,
            split="test"
        )
```

## Integration with Dataclasses

DataModules use DataLoaders to convert dataset items into batches:

```python
def train_dataloader(self) -> DataLoader:
    """Create the train dataloader."""
    return DataLoader(
        dataset=self.train_dataset,
        batch_size=self.train_batch_size,
        shuffle=True,
        num_workers=self.num_workers,
        collate_fn=ImageBatch.collate    # Converts list of ImageItems to ImageBatch
    )
```

The data flow is:

1. Dataset returns {doc}`ImageItem <./dataclasses>` objects
2. DataLoader collates them into {doc}`ImageBatch <./dataclasses>` objects
3. Model receives ImageBatch for training/inference

## Example DataModules

### 1. Image DataModule

```python
from anomalib.data import MVTecAD

datamodule = MVTecAD(
    root="./datasets/MVTecAD",
    category="bottle",
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=8
)

# Setup creates the datasets
datamodule.setup()

# Get train dataloader
train_loader = datamodule.train_dataloader()

# Access batches
for batch in train_loader:
    print(batch.image.shape)      # torch.Size([32, 3, 256, 256])
    print(batch.gt_label.shape)   # torch.Size([32])
```

### 2. Video DataModule

```python
from anomalib.data import Avenue

datamodule = Avenue(
    clip_length_in_frames=2,
    frames_between_clips=1,
    target_frame="last",
)
datamodule.setup()
i, data = next(enumerate(datamodule.train_dataloader()))
data["image"].shape
# torch.Size([32, 2, 3, 256, 256])
```

### 3. Depth DataModule

```python
from anomalib.data import MVTec3D

datamodule = MVTec3D(
    root="./datasets/MVTec3D",
    category="bagel",
    train_batch_size=32,
)

# Access RGB-D batches
i, data = next(enumerate(datamodule.train_dataloader()))
data["image"].shape
# torch.Size([32, 3, 256, 256])
data["depth_map"].shape
# torch.Size([32, 1, 256, 256])
```

## Creating Custom DataModules

To create a custom DataModule:

```python
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from anomalib.data.dataclasses import ImageBatch

class CustomDataModule(LightningDataModule):
    def __init__(
        self,
        root: str,
        category: str,
        train_batch_size: int = 32,
        **kwargs
    ):
        super().__init__()
        self.root = root
        self.category = category
        self.image_size = image_size
        self.train_batch_size = train_batch_size

    def setup(self, stage: str | None = None):
        """Initialize datasets."""
        if stage == "fit" or stage is None:
            self.train_dataset = CustomDataset(
                root=self.root,
                category=self.category,
                split="train"
            )

    def train_dataloader(self) -> DataLoader:
        """Create train dataloader."""
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=ImageBatch.collate
        )
```

## Best Practices

1. **Data Organization**:

   - Keep dataset creation in `setup()`
   - Use appropriate batch sizes for train/eval
   - Handle multi-GPU scenarios

2. **Memory Management**:

   - Use appropriate number of workers
   - Clear cache between epochs if needed
   - Handle GPU memory efficiently

3. **Transforms**:

   - Apply consistent transforms across splits
   - Use torchvision.transforms.v2
   - Handle different input modalities

4. **Validation**:
   - Verify data shapes and types
   - Check batch size consistency
   - Validate paths and parameters

```{seealso}
- For details on dataset implementation, see the {doc}`datasets guide <./datasets>`.
- For information about the data objects, see the {doc}`dataclasses guide <./dataclasses>`.
```
