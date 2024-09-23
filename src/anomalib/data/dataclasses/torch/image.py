"""Torch-based dataclasses for image data in Anomalib.

This module provides PyTorch-based implementations of the generic dataclasses
used in Anomalib for image data. These classes are designed to work with PyTorch
tensors for efficient data handling and processing in anomaly detection tasks.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torchvision.tv_tensors import Image, Mask

from anomalib.data.dataclasses.generic import BatchIterateMixin, _ImageInputFields
from anomalib.data.dataclasses.numpy.image import NumpyImageBatch, NumpyImageItem
from anomalib.data.dataclasses.torch.base import Batch, DatasetItem, ToNumpyMixin
from anomalib.data.validators.torch.image import ImageBatchValidator, ImageValidator


@dataclass
class ImageItem(
    ToNumpyMixin[NumpyImageItem],
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

    @staticmethod
    def validate_image(image: torch.Tensor) -> torch.Tensor:
        return ImageValidator.validate_image(image)

    @staticmethod
    def validate_gt_label(gt_label: torch.Tensor | int | None) -> torch.Tensor | None:
        return ImageValidator.validate_gt_label(gt_label)

    @staticmethod
    def validate_gt_mask(gt_mask: torch.Tensor | None) -> Mask | None:
        return ImageValidator.validate_gt_mask(gt_mask)

    @staticmethod
    def validate_mask_path(mask_path: str | None) -> str | None:
        return ImageValidator.validate_mask_path(mask_path)

    @staticmethod
    def validate_anomaly_map(anomaly_map: torch.Tensor | None) -> Mask | None:
        return ImageValidator.validate_anomaly_map(anomaly_map)

    @staticmethod
    def validate_pred_score(pred_score: torch.Tensor | np.ndarray | float | None) -> torch.Tensor | None:
        return ImageValidator.validate_pred_score(pred_score)

    @staticmethod
    def validate_pred_mask(pred_mask: torch.Tensor | None) -> torch.Tensor | None:
        return ImageValidator.validate_pred_mask(pred_mask)

    @staticmethod
    def validate_pred_label(pred_label: torch.Tensor | np.ndarray | float | None) -> torch.Tensor | None:
        return ImageValidator.validate_pred_label(pred_label)

    @staticmethod
    def validate_image_path(image_path: str | None) -> str | None:
        return ImageValidator.validate_image_path(image_path)


@dataclass
class ImageBatch(
    ToNumpyMixin[NumpyImageBatch],
    BatchIterateMixin[ImageItem],
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

    @staticmethod
    def validate_image(image: Image) -> Image:
        return ImageBatchValidator.validate_image(image)

    def validate_gt_label(self, gt_label: torch.Tensor | Sequence[int] | None) -> torch.Tensor | None:
        return ImageBatchValidator.validate_gt_label(gt_label)

    def validate_gt_mask(self, gt_mask: Mask | None) -> Mask | None:
        return ImageBatchValidator.validate_gt_mask(gt_mask)

    def validate_mask_path(self, mask_path: Sequence[str] | Sequence[str] | None) -> list[str] | None:
        return ImageBatchValidator.validate_mask_path(mask_path)

    def validate_anomaly_map(self, anomaly_map: torch.Tensor | np.ndarray | None) -> torch.Tensor | None:
        return ImageBatchValidator.validate_anomaly_map(anomaly_map)

    def validate_pred_score(self, pred_score: torch.Tensor | None) -> torch.Tensor | None:
        return ImageBatchValidator.validate_pred_score(pred_score, self.anomaly_map)

    def validate_pred_mask(self, pred_mask: torch.Tensor | None) -> torch.Tensor | None:
        return ImageBatchValidator.validate_pred_mask(pred_mask)

    @staticmethod
    def validate_pred_label(pred_label: torch.Tensor | None) -> torch.Tensor | None:
        return ImageBatchValidator.validate_pred_label(pred_label)

    @staticmethod
    def validate_image_path(image_path: list[str]) -> list[str] | None:
        return ImageBatchValidator.validate_image_path(image_path)
