"""Torch-based dataclasses for image data in Anomalib.

This module provides PyTorch-based implementations of the generic dataclasses
used in Anomalib for image data. These classes are designed to work with PyTorch
tensors for efficient data handling and processing in anomaly detection tasks.
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
    """Dataclass for individual image items in Anomalib datasets using PyTorch tensors.

    This class combines the functionality of ToNumpyMixin, _ImageInputFields, and
    DatasetItem to represent single image data points in Anomalib. It includes
    image-specific fields and provides methods for data validation and conversion
    to numpy format.

    The class is designed to work with PyTorch tensors and includes fields for
    the image data, ground truth labels and masks, anomaly maps, and related metadata.

    Attributes:
        Inherited from _ImageInputFields and DatasetItem.

    Methods:
        Inherited from ToNumpyMixin, including to_numpy() for conversion to numpy format.

    Examples:
        >>> item = ImageItem(
        ...     image=torch.rand(3, 224, 224),
        ...     gt_label=torch.tensor(1),
        ...     gt_mask=torch.rand(224, 224) > 0.5,
        ...     image_path="path/to/image.jpg"
        ... )

        >>> print(item.image.shape)
        torch.Size([3, 224, 224])

        >>> numpy_item = item.to_numpy()
        >>> print(type(numpy_item))
        <class 'anomalib.dataclasses.numpy.NumpyImageItem'>
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
    """Dataclass for batches of image items in Anomalib datasets using PyTorch tensors.

    This class combines the functionality of ``ToNumpyMixin``, ``BatchIterateMixin``,
    ``_ImageInputFields``, and ``Batch`` to represent collections of image data points in Anomalib.
    It includes image-specific fields and provides methods for batch operations,
    iteration over individual items, and conversion to numpy format.

    The class is designed to work with PyTorch tensors and includes fields for
    batches of image data, ground truth labels and masks, anomaly maps, and related metadata.

    Examples:
        >>> batch = ImageBatch(
        ...     image=torch.rand(32, 3, 224, 224),
        ...     gt_label=torch.randint(0, 2, (32,)),
        ...     gt_mask=torch.rand(32, 224, 224) > 0.5,
        ...     image_path=["path/to/image_{}.jpg".format(i) for i in range(32)]
        ... )

        >>> print(batch.image.shape)
        torch.Size([32, 3, 224, 224])

        >>> for item in batch:
        ...     print(item.image.shape)
        torch.Size([3, 224, 224])

        >>> numpy_batch = batch.to_numpy()
        >>> print(type(numpy_batch))
        <class 'anomalib.dataclasses.numpy.NumpyImageBatch'>
    """

    item_class = ImageItem
    numpy_class = NumpyImageBatch
