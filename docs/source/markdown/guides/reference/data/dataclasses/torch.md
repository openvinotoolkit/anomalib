# Torch Dataclasses

The torch dataclasses module provides PyTorch-based implementations of the generic dataclasses used in Anomalib. These classes are designed to work with PyTorch tensors for efficient data handling and processing in anomaly detection tasks.

```{eval-rst}
.. currentmodule:: anomalib.data.dataclasses.torch
```

## Overview

The module includes several categories of dataclasses:

- **Base Classes**: Generic PyTorch-based data structures
- **Image Classes**: Specialized for image data processing
- **Video Classes**: Designed for video data handling
- **Depth Classes**: Specific to depth-based anomaly detection

## Base Classes

### DatasetItem

```{eval-rst}
.. autoclass:: DatasetItem
   :members:
   :show-inheritance:
```

### Batch

```{eval-rst}
.. autoclass:: Batch
   :members:
   :show-inheritance:
```

### InferenceBatch

```{eval-rst}
.. autoclass:: InferenceBatch
   :members:
   :show-inheritance:
```

### ToNumpyMixin

```{eval-rst}
.. autoclass:: ToNumpyMixin
   :members:
   :show-inheritance:
```

## Image Classes

### ImageItem

```{eval-rst}
.. autoclass:: ImageItem
   :members:
   :show-inheritance:
```

### ImageBatch

```{eval-rst}
.. autoclass:: ImageBatch
   :members:
   :show-inheritance:
```

## GRD-Net Image Classes

`GRDNetItem` and `GRDNetBatch` extend the standard image types with training-only
`roi_mask` and `roi_mask_path` fields. Masks have shape `[H, W]` for items and
`[B, H, W]` for batches. Conversion to NumPy returns the standard image dataclasses
without ROI fields. See {doc}`../datamodules/image/grdnet` for the aligned folder pipeline.

```{eval-rst}
.. autoclass:: GRDNetItem
   :members: to_numpy
   :show-inheritance:

.. autoclass:: GRDNetBatch
   :members: to_numpy
   :show-inheritance:
```

## Video Classes

### VideoItem

```{eval-rst}
.. autoclass:: VideoItem
   :members:
   :show-inheritance:
```

### VideoBatch

```{eval-rst}
.. autoclass:: VideoBatch
   :members:
   :show-inheritance:
```

## Depth Classes

### DepthItem

```{eval-rst}
.. autoclass:: DepthItem
   :members:
   :show-inheritance:
```

### DepthBatch

```{eval-rst}
.. autoclass:: DepthBatch
   :members:
   :show-inheritance:
```

## See Also

- {doc}`./index`
- {doc}`./numpy`
