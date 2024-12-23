```{eval-rst}
:orphan:
```

# Working with Anomalib Dataclasses

This guide explains how to use the dataclasses in Anomalib, from basic usage to advanced use cases across different modalities.

## Basic Concepts

Anomalib uses dataclasses to represent and validate data throughout the pipeline. The dataclasses are designed to be:

- **Type-safe**: All fields are validated to ensure correct types and shapes
- **Modality-specific**: Specialized classes for images, videos, and depth data
- **Framework-agnostic**: Support for both PyTorch and NumPy backends
- **Batch-aware**: Handle both single items and batches of data

The dataclass system is built around two main concepts:

1. **Item**: Single data instance (e.g., one image)
2. **Batch**: Collection of items with batch processing capabilities

## Core Components

The dataclass system consists of several key components:

1. **Input Fields**: Base fields for anomaly detection data

   - `image`: Input image/video
   - `gt_label`: Ground truth label
   - `gt_mask`: Ground truth segmentation mask
   - `mask_path`: Path to mask file

2. **Output Fields**: Fields for model predictions
   - `anomaly_map`: Predicted anomaly heatmap
   - `pred_score`: Predicted anomaly score
   - `pred_mask`: Predicted segmentation mask
   - `pred_label`: Predicted label
   - `explanation`: Path to explanation visualization

## Working with Items

Here's a simple example using the image dataclass with PyTorch:

```{code-block} python
from anomalib.data.dataclasses.torch import ImageItem
import torch

# Create a single image item
item = ImageItem(
    image=torch.rand(3, 224, 224),      # RGB image
    gt_label=torch.tensor(0),           # Normal (0) or anomalous (1)
    image_path="path/to/image.jpg"      # Path to source image
)

# Access the data
print(item.image.shape)         # torch.Size([3, 224, 224])
print(item.gt_label.item())     # 0
```

## Working with Batches

Anomalib provides powerful batch classes for efficient data processing. A batch is a collection of items with additional functionality for batch operations.

### Creating Batches

There are several ways to create batches:

```{code-block} python
from anomalib.data.dataclasses.torch import ImageBatch, ImageItem
import torch

# 1. Direct creation with tensors
batch = ImageBatch(
    image=torch.rand(32, 3, 224, 224),          # Batch of 32 images
    gt_label=torch.randint(0, 2, (32,)),        # Batch of labels
    image_path=[f"image_{i}.jpg" for i in range(32)]
)

# 2. From a list of items
items = [
    ImageItem(
        image=torch.rand(3, 224, 224),
        gt_label=torch.tensor(0),
        image_path=f"image_{i}.jpg"
    )
    for i in range(32)
]

batch = ImageBatch.collate(items)
```

### Batch Properties

Batches provide several useful properties:

```{code-block} python
# Get batch size
print(batch.batch_size)    # 32

# Get device information
print(batch.image.device)        # cuda:0 or cpu

# Get shape information
print(batch.image.shape)   # torch.Size([32, 3, 224, 224])
```

### Iterating Over Batches

Batches can be iterated to access individual items:

```{code-block} python
# Simple iteration
for item in batch:
    print(item.image.shape)    # torch.Size([3, 224, 224])
    # Process the input

# Enumerate for index access
for idx, _item in enumerate(batch):
    print(f"Processing item {idx}")
    # Process the input
```

### Batch Operations

Batches support various operations:

```{code-block} python
# Update batch fields
batch.update(pred_scores=torch.rand(32))

# Split batch into items
items = batch.items
```

### Batch Validation

Batches automatically validate:

- Consistent batch size across all fields
- Compatible tensor shapes
- Device consistency
- Data type compatibility

```{code-block} python
# This will raise a ValidationError due to float labels (gt_label must be int or boolean)
invalid_batch = ImageBatch(
    image=torch.rand(32, 3, 224, 224),
    gt_label=torch.randint(0.0, 1.0, (32,))  # float labels
)
```

## Different Modalities

### 1. Image Data

The most basic form, supporting RGB images:

```{code-block} python
from anomalib.data.dataclasses.torch import ImageItem, ImageBatch

# Single image
item = ImageItem(
    image=torch.rand(3, 224, 224),
    gt_label=torch.tensor(0),
    image_path="image.jpg"
)

# Batch of images
batch = ImageBatch(
    image=torch.rand(32, 3, 224, 224),
    gt_label=torch.randint(0, 2, (32,)),
    image_path=[f"image_{i}.jpg" for i in range(32)]
)
```

### 2. Video Data

For video processing with temporal information:

```{code-block} python
from anomalib.data.dataclasses.torch import VideoItem, VideoBatch

# Single video item
item = VideoItem(
    image=torch.rand(10, 3, 224, 224),  # 10 frames
    gt_label=torch.tensor(0),
    video_path="path/to/video.mp4",
)

# Batch of video items
batch = VideoBatch(
    image=torch.rand(32, 10, 3, 224, 224),  # 32 videos, 10 frames
    gt_label=torch.randint(0, 2, (32,)),
    video_path=["video_{}.mp4".format(i) for i in range(32)],
)
```

### 3. Depth Data

For RGB-D or depth-only processing:

```{code-block} python
from anomalib.data.dataclasses.torch import DepthItem, DepthBatch

# Single depth item
item = DepthItem(
    image=torch.rand(3, 224, 224),          # RGB image
    depth_map=torch.rand(224, 224),         # Depth map
    image_path="rgb.jpg",
    depth_path="depth.png",
)

# Batch of depth items
batch = DepthBatch(
    image=torch.rand(32, 3, 224, 224),           # RGB images
    depth_map=torch.rand(32, 224, 224),          # Depth maps
    image_path=[f"rgb_{i}.jpg" for i in range(32)],
    depth_path=[f"depth_{i}.png" for i in range(32)],
)
```

## Advanced Features

### 1. Converting Between Frameworks

All dataclasses support conversion between PyTorch and NumPy:

```{code-block} python
# Items
numpy_item = torch_item.to_numpy()

# Batches
numpy_batch = torch_batch.to_numpy()
```

### 2. Validation

Dataclasses automatically validate inputs:

- Correct tensor shapes and dimensions
- Compatible data types
- File path existence (optional)
- Batch size consistency
- Device consistency within batches

### 3. Updating Fields

In-place or copy updates are supported for both items and batches:

```{code-block} python
# Items
item.update(pred_score=0.8)
new_item = item.update(in_place=False, pred_label=1)

# Batches
batch.update(pred_scores=torch.rand(32))
new_batch = batch.update(in_place=False, pred_labels=torch.randint(0, 2, (32,)))
```

## Best Practices

1. **Type Hints**: Always use appropriate type hints when subclassing
2. **Validation**: Implement custom validators for special requirements
3. **Batch Size**: Keep batch dimensions consistent across all fields
4. **Memory**: Use appropriate data types (uint8 for images, float32 for features)
5. **Paths**: Use relative paths when possible for portability
6. **Batch Processing**: Use batch operations when possible for better performance
7. **Device Management**: Keep tensors on the same device within a batch

## Common Pitfalls

1. **Inconsistent Shapes**: Ensure all batch dimensions match
2. **Missing Fields**: Required fields must be provided
3. **Type Mismatches**: Use correct tensor types (torch vs numpy)
4. **Memory Leaks**: Clear large batches when no longer needed
5. **Path Issues**: Use proper path separators for cross-platform compatibility
6. **Device Mismatches**: Ensure all tensors in a batch are on the same device
7. **Batch Size Inconsistency**: Maintain consistent batch sizes across all fields
