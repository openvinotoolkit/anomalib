```{eval-rst}
:orphan:
```

# Dataclasses

This guide explains how to use the dataclasses in Anomalib, from basic usage to advanced use cases across different modalities.

## Basic Concepts

Anomalib uses dataclasses to represent and validate data throughout the pipeline. Anomalib's dataclasses are based on python's
native dataclasses, but are extended with several useful features to facilitate input validation and easy conversion.
Dataclasses are used by the `AnomalibDataset` and `AnomalibDatamodule` to represent input data and ground truth annotations,
and by the `AnomalibModule` to store the model predictions. For basic users, knowing how to access and update the fields
of Anomalib's dataclasses is sufficient to cover most use-cases.

The dataclasses are designed to be:

- **Type-safe**: All fields are validated to ensure correct types and shapes
- **Modality-specific**: Specialized classes for images, videos, and depth data
- **Framework-specific**: Support for both PyTorch and NumPy backends
- **Batch-aware**: Handle both single items and batches of data

The dataclass system is built around two main concepts:

1. **Item**: Single data instance (e.g., one image)
2. **Batch**: Collection of items with batch processing capabilities

The Item and Batch classes are defined separately for the different data modalities in the libary. For example, when
working with image data, the relevant classes are `ImageItem` and `ImageBatch`.

## Input- and Output fields

All dataclasses are equipped with the following standard data fields:

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

Out of these standard fields, only `image` is mandatory. All other fields are optional. In addition to the standard fields,
Anomalib's dataclasses may contain additional modality-specific input and output fields, depending on the modality of the
data (Image, Video, Depth).

## Basic Usage

### Creating a dataclass instance

To create a new dataclass instance, simply pass the data to the constructor using the keyword arguments. For example, we could use the following code to create a new instance of an `ImageItem` from a randomly generated image and an all-negative (no anomalous pixels) ground truth mask.

```{code-block} python
import torch
from anomalib.data import ImageItem

item = ImageItem(
    image=torch.rand((3, 224, 224)),
    image_path="path/to/my/image.png"
    gt_label=0,
    gt_mask=torch.zeros((224, 224)),
)
```

After creating the instance, you can directly access any of the provided fields of the dataclass

```{code-block} python
print(item.image_path)   # "path/to/my/image.png"
print(item.image.shape)  # torch.Size([3, 224, 224])
```

Similarly, we could create a batch of images, by directly defining an image tensor with a leading batch dimension. Let's create a random batch consisting of 8 images.

```{code-block} python
import torch
from anomalib.data import ImageItem

batch = ImageBatch(
    image=torch.rand((8, 3, 224, 224)),
    gt_label=[0, ] * 8,
    gt_mask=torch.zeros((8, 224, 224)),
)
```

Again, we can inspect the fields of the batch instance by accessing them directly. In addition, the `Batch` class provides
a useful `batch_size` property to quickly retrieve the number of items in the batch.

```{code-block} python
print(batch.image.shape)  # torch.Size([8, 3, 224, 224])
print(batch.batch_size)   # 8
```

> **Note:**
> The above examples are for illustrative purposes. In general, most use-cases don't require instantiating dataclasses explicitly,
> as Anomalib's modules create and return the dataclass instances.

### Validation and formatting

The dataclass performs some validation checks to assert that the provided values have the expected shape and format, and
automatically converts the values to the correct datatype where possible. This ensures that all instances of the dataclass
will always use the same shapes and data types to represent the input- and output fields!

```{code-block} python
item = ImageItem(
    image=torch.rand((8, 3, 224, 224))
)
# raises ValueError because provided value has one dimension too many (batch cannot be converted to single item).

batch = ImageBatch(
    image=torch.rand((3, 224, 224)),
    gt_label = [1],
)
print(batch.image.shape)  # torch.Size([1, 3, 224, 224])  <-- leading batch dimension added automatically
print(batch.gt_label)     # tensor([True])                 <-- positive label converted to boolean tensor
```

### Updating a dataclass instance

To update a field of a dataclass instance, simply overwrite its value. The dataclass will automatically run the validation
checks before assigning the updated value to the instance!

```{code-block} python
item = ImageItem(
    image=torch.rand((3, 224, 224)),
    gt_label=tensor(False),
)

# overwrite an existing field
item.gt_label = tensor(True)
print(item.gt_label)  # tensor(True)

# assign a previously unassigned field
item.image_path = "path/to/my/image.png"
print(item.image_path)  # "path/to/my/image.png"

# input validation and auto formatting still works
item.pred_score = 0.45
print(item.pred_score)  # tensor(0.4500)
```

As an alternative method of updating dataclass fields, Anomalib's dataclasses are equipped with the `update` method. By default,
the `update` method updates the dataclass instance inplace, meaning that the original instance will be modified.

```{code-block} python
item = ImageItem(
    image=torch.rand((3, 224, 224)),
    pred_score=0.33,
)
item.update(pred_score=0.87)  # this is equivalent to item.pred_score=0.87
print(item.pred_score)        # 0.87
```

If you want to keep the original item, you can pass `inplace=False`, and use the new instance returned by `update`.

```{code-block} python
item = ImageItem(
    image=torch.rand((3, 224, 224)),
    pred_score=0.33,
)
new_item = item.update(pred_score=0.87, inplace=False)  # the original item will remain unmodified
print(item.pred_score)      # 0.33
print(new_item.pred_score)  # 0.87
```

The `update` method can be useful in situations where you want to update multiple fields at once, for example from a dictionary
of predictions returned by your model. This can be achieved by specifying each field as a keyword argument, or by passing an
entire dictionary using the `**` notation:

```{code-block} python
item.update(
    pred_score=0.87,
    pred_label=True,
)

# the following would have the same effect as the statement above
predictions = {
    "pred_score": 0.87,
    "pred_label": True,
}
item.update(**predictions)
```

### Converting between items and batch

It is very easy to switch between `Item` and `Batch` instances. To separate a `Batch` instance into a list of `Item`s, simply
use the `items` property:

```{code-block} python
batch = ImageBatch(
    image=torch.rand((4, 3, 360, 240))
)
items = batch.items  # list of Item instances
```

Conversely, `Batch` has a `collate` method that can be invoked to create a new `Batch` instance from a list of `Item`s.

```{code-block} python
new_batch = ImageBatch.collate(items)  # construct a new batch from a list of Items
```

It is also possible to directly iterate over the `Item`s in a batch, without explicitly calling the `items` property:

```{code-block} python
for item in batch:
    # use item
```

### Converting Between Frameworks

All dataclasses support conversion between PyTorch and NumPy:

```{code-block} python
# Items
numpy_item = torch_item.to_numpy()

# Batches
numpy_batch = torch_batch.to_numpy()
```

## Supported Modalities

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

## Best Practices

1. **Type Hints**: Always use appropriate type hints when subclassing
2. **Validation**: Implement custom validators for special requirements
3. **Batch Size**: Keep batch dimensions consistent across all fields
4. **Paths**: Use relative paths when possible for portability
5. **Batch Processing**: Use batch operations when possible for better performance
6. **Device Management**: Keep tensors on the same device within a batch

## Common Pitfalls

1. **Inconsistent Shapes**: Ensure all batch dimensions match
2. **Missing Fields**: Required fields must be provided
3. **Type Mismatches**: Use correct tensor types (torch vs numpy)
4. **Memory Leaks**: Clear large batches when no longer needed
5. **Path Issues**: Use proper path separators for cross-platform compatibility
6. **Device Mismatches**: Ensure all tensors in a batch are on the same device
7. **Batch Size Inconsistency**: Maintain consistent batch sizes across all fields
