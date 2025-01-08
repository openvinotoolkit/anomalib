"""Torch-based dataclasses for image data in Anomalib.

This module provides PyTorch-based implementations of the generic dataclasses
used in Anomalib for image data. These classes are designed to work with PyTorch
tensors for efficient data handling and processing in anomaly detection tasks.

The module contains two main classes:
    - :class:`ImageItem`: For single image data items
    - :class:`ImageBatch`: For batched image data items

Example:
    Create and use a torch image item::

        >>> from anomalib.data.dataclasses.torch import ImageItem
        >>> import torch
        >>> item = ImageItem(
        ...     image=torch.rand(3, 224, 224),
        ...     gt_label=torch.tensor(0),
        ...     image_path="path/to/image.jpg"
        ... )
        >>> item.image.shape
        torch.Size([3, 224, 224])
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from torchvision.tv_tensors import Image

from anomalib.data.dataclasses.generic import BatchIterateMixin, _ImageInputFields
from anomalib.data.dataclasses.numpy.image import NumpyImageBatch, NumpyImageItem
from anomalib.data.dataclasses.torch.base import Batch, DatasetItem, ToNumpyMixin
from anomalib.data.validators.torch.image import ImageBatchValidator, ImageValidator


@dataclass
class ImageItem(
    ToNumpyMixin[NumpyImageItem],
    ImageValidator,
    _ImageInputFields[str],
    DatasetItem[Image],
):
    """Dataclass for individual image items in Anomalib datasets using PyTorch.

    This class combines :class:`_ImageInputFields` and :class:`DatasetItem` for
    image-based anomaly detection. It includes image-specific fields and validation
    methods to ensure proper formatting for Anomalib's image-based models.

    The class uses the following type parameters:
        - Image: :class:`torch.Tensor` with shape ``(C, H, W)``
        - Label: :class:`torch.Tensor`
        - Mask: :class:`torch.Tensor` with shape ``(H, W)``
        - Path: :class:`str`

    Example:
        >>> import torch
        >>> from anomalib.data.dataclasses.torch import ImageItem
        >>> item = ImageItem(
        ...     image=torch.rand(3, 224, 224),
        ...     gt_label=torch.tensor(0),
        ...     image_path="path/to/image.jpg"
        ... )
        >>> item.image.shape
        torch.Size([3, 224, 224])

        Convert to numpy format:
        >>> numpy_item = item.to_numpy()
        >>> type(numpy_item).__name__
        'NumpyImageItem'
    """

    numpy_class = NumpyImageItem


@dataclass
class ImageBatch(
    ToNumpyMixin[NumpyImageBatch],
    BatchIterateMixin[ImageItem],
    ImageBatchValidator,
    _ImageInputFields[list[str]],
    Batch[Image],
):
    """Dataclass for batches of image items in Anomalib datasets using PyTorch.

    This class combines :class:`_ImageInputFields` and :class:`Batch` for batches
    of image data. It includes image-specific fields and methods for batch
    operations and iteration.

    The class uses the following type parameters:
        - Image: :class:`torch.Tensor` with shape ``(B, C, H, W)``
        - Label: :class:`torch.Tensor` with shape ``(B,)``
        - Mask: :class:`torch.Tensor` with shape ``(B, H, W)``
        - Path: :class:`list` of :class:`str`

    Where ``B`` represents the batch dimension.

    Example:
        >>> import torch
        >>> from anomalib.data.dataclasses.torch import ImageBatch
        >>> batch = ImageBatch(
        ...     image=torch.rand(32, 3, 224, 224),
        ...     gt_label=torch.randint(0, 2, (32,)),
        ...     image_path=[f"path/to/image_{i}.jpg" for i in range(32)]
        ... )
        >>> batch.image.shape
        torch.Size([32, 3, 224, 224])

        Iterate over batch:
        >>> for item in batch:
        ...     assert item.image.shape == torch.Size([3, 224, 224])

        Convert to numpy format:
        >>> numpy_batch = batch.to_numpy()
        >>> type(numpy_batch).__name__
        'NumpyImageBatch'
    """

    item_class = ImageItem
    numpy_class = NumpyImageBatch
