```{eval-rst}
:orphan:
```

# Datasets

This guide explains how datasets work in Anomalib, from the base implementation to specific dataset types and how to create your own dataset.

## Base Dataset Structure

Anomalib's dataset system is built on top of PyTorch's `Dataset` class and uses pandas DataFrames to manage dataset samples. The base class `AnomalibDataset` provides the foundation for all dataset implementations.

### Core Components

The dataset consists of three main components:

1. **Samples DataFrame**: The heart of each dataset is a DataFrame containing:

   - `image_path`: Path to the image file
   - `split`: Dataset split (train/test/val)
   - `label_index`: Label index (0 for normal, 1 for anomalous)
   - `mask_path`: Path to mask file (for segmentation tasks)

   Example DataFrame:

   ```python
   df = pd.DataFrame({
       'image_path': ['path/to/image.png'],
       'label': ['anomalous'],
       'label_index': [1],
       'mask_path': ['path/to/mask.png'],
       'split': ['train']
   })
   ```

2. **Transforms**: Optional transformations applied to images

3. **Task Type**: Classification or Segmentation

## Dataset Types

Anomalib supports different types of datasets based on modality:

### 1. Image Datasets

The most common type, supporting RGB images:

```python
from anomalib.data.datasets import MVTecADDataset

# Create MVTecAD dataset
dataset = MVTecADDataset(
    root="./datasets/MVTecAD",
    category="bottle",
    split="train"
)

# Access an item
item = dataset[0]
print(item.image.shape)         # RGB image
print(item.gt_label.item())     # Label (0 or 1)
print(item.gt_mask.shape)       # Segmentation mask (if available)
```

### 2. Video Datasets

For video anomaly detection:

```python
from anomalib.data.datasets import Avenue

# Create video dataset
dataset = AvenueDataset(
    root="./datasets/avenue",
    split="test",
    transform=transform
)

# Access an item
item = dataset[0]
print(item.frames.shape)        # Video frames
print(item.target_frame)        # Frame number
```

### 3. Depth Datasets

For RGB-D or depth-only data:

```python
from anomalib.data.datasets import MVTec3DDataset

# Create depth dataset
dataset = MVTec3DDataset(
    root="./datasets/MVTec3D",
    category="bagel",
    split="train",
)

# Access an item
item = dataset[0]
print(item.image.shape)         # RGB image
print(item.depth_map.shape)     # Depth map
```

## Dataset Loading Process

The dataset loading process follows these steps:

1. **Initialization**:

   ```python
   def __init__(self, transform=None):
       self.transform = transform
       self._samples = None
       self._category = None
   ```

2. **Sample Collection**:

   ```python
   @property
   def samples(self):
       if self._samples is None:
           raise RuntimeError("Samples DataFrame not set")
       return self._samples
   ```

3. **Item Loading**:

   ```python
   def __getitem__(self, index):
       sample = self.samples.iloc[index]
       image = read_image(sample.image_path)

       if self.transform:
           image = self.transform(image)

       return ImageItem(
           image=image,
           gt_label=sample.label_index
       )
   ```

### Integration with Dataclasses

Anomalib datasets are designed to work seamlessly with the dataclass system. When you access items from a dataset:

- Single items are returned as {doc}`Item objects <./dataclasses>` (e.g., `ImageItem`, `VideoItem`, `DepthItem`)
- When used with PyTorch's DataLoader, items are automatically collated into {doc}`Batch objects <./dataclasses>` (e.g., `ImageBatch`, `VideoBatch`, `DepthBatch`)

For example:

```python
# Single item access returns an Item object
item = dataset[0]              # Returns ImageItem

# DataLoader automatically creates Batch objects
dataloader = DataLoader(dataset, batch_size=32)
batch = next(iter(dataloader))  # Returns ImageBatch
```

```{seealso}
For more details on working with Item and Batch objects, see the {doc}`dataclasses guide <./dataclasses>`.
```

## Creating Custom Datasets

To create a custom dataset, extend the `AnomalibDataset` class:

```python
from anomalib.data.datasets.base import AnomalibDataset
from pathlib import Path
import pandas as pd

class CustomDataset(AnomalibDataset):
    """Custom dataset implementation."""

    def __init__(
        self,
        root: Path | str = "./datasets/Custom",
        category: str = "default",
        transform = None,
        split = None,
    ):
        super().__init__(transform=transform)

        # Set up dataset
        self.root = Path(root)
        self.category = category
        self.split = split

        # Create samples DataFrame
        self.samples = self._make_dataset()

    def _make_dataset(self) -> pd.DataFrame:
        """Create dataset samples DataFrame."""
        samples_list = []

        # Collect normal samples
        normal_path = self.root / "normal"
        for image_path in normal_path.glob("*.png"):
            samples_list.append({
                "image_path": str(image_path),
                "label": "normal",
                "label_index": 0,
                "split": "train"
            })

        # Collect anomalous samples
        anomaly_path = self.root / "anomaly"
        for image_path in anomaly_path.glob("*.png"):
            mask_path = anomaly_path / "masks" / f"{image_path.stem}_mask.png"
            samples_list.append({
                "image_path": str(image_path),
                "label": "anomaly",
                "label_index": 1,
                "mask_path": str(mask_path),
                "split": "test"
            })

        # Create DataFrame
        samples = pd.DataFrame(samples_list)
        samples.attrs["task"] = "segmentation"

        return samples
```

### Expected Directory Structure

For the custom dataset above:

```bash
datasets/
└── Custom/
    ├── normal/
    │   ├── 001.png
    │   ├── 002.png
    │   └── ...
    └── anomaly/
        ├── 001.png
        ├── 002.png
        └── masks/
            ├── 001_mask.png
            ├── 002_mask.png
            └── ...
```

## Best Practices

1. **Data Organization**:

   - Keep consistent directory structure
   - Use clear naming conventions
   - Separate train/test splits

2. **Validation**:

   - Validate image paths exist
   - Ensure mask-image correspondence
   - Check label consistency

3. **Performance**:

   - Use appropriate data types
   - Implement efficient data loading
   - Cache frequently accessed data

4. **Error Handling**:
   - Provide clear error messages
   - Handle missing files gracefully
   - Validate input parameters

## Common Pitfalls

1. **Path Issues**:

   - Incorrect root directory
   - Missing mask files
   - Inconsistent file extensions

2. **Data Consistency**:

   - Mismatched image-mask pairs
   - Inconsistent image sizes
   - Wrong label assignments

3. **Memory Management**:

   - Loading too many images at once
   - Not releasing unused resources
   - Inefficient data structures

4. **Transform Issues**:
   - Incompatible transforms
   - Missing normalization
   - Incorrect transform order
