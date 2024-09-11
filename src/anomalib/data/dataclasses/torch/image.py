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
from anomalib.data.validators.torch.image import ImageValidator


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

    def _validate_image(self, image: torch.Tensor) -> torch.Tensor:
        return ImageValidator.validate_image(image)

    def _validate_gt_label(self, gt_label: torch.Tensor | int | None) -> torch.Tensor | None:
        return ImageValidator.validate_gt_label(gt_label)

    def _validate_gt_mask(self, gt_mask: torch.Tensor | None) -> Mask | None:
        return ImageValidator.validate_gt_mask(gt_mask)

    def _validate_mask_path(self, mask_path: str | None) -> str | None:
        return ImageValidator.validate_mask_path(mask_path)

    def _validate_anomaly_map(self, anomaly_map: torch.Tensor | None) -> Mask | None:
        return ImageValidator.validate_anomaly_map(anomaly_map)

    def _validate_pred_score(self, pred_score: torch.Tensor | np.ndarray | None) -> torch.Tensor | None:
        return ImageValidator.validate_pred_score(pred_score, self.anomaly_map)

    def _validate_pred_mask(self, pred_mask: torch.Tensor | None) -> Mask | None:
        return ImageValidator.validate_pred_mask(pred_mask)

    def _validate_pred_label(self, pred_label: torch.Tensor | np.ndarray | None) -> torch.Tensor | None:
        return ImageValidator.validate_pred_label(pred_label)

    def _validate_image_path(self, image_path: str | None) -> str | None:
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

    def _validate_image(self, image: Image) -> Image:
        assert isinstance(image, torch.Tensor), f"Image must be a torch.Tensor, got {type(image)}."
        assert image.ndim in {3, 4}, f"Image must have shape [C, H, W] or [N, C, H, W], got shape {image.shape}."
        if image.ndim == 3:
            image = image.unsqueeze(0)  # add batch dimension
        assert image.shape[1] == 3, f"Image must have 3 channels, got {image.shape[0]}."
        return Image(image, dtype=torch.float32)

    def _validate_gt_label(self, gt_label: torch.Tensor | Sequence[int] | None) -> torch.Tensor:
        if gt_label is None:
            return None
        if isinstance(gt_label, Sequence):
            gt_label = torch.tensor(gt_label)
        assert isinstance(
            gt_label,
            torch.Tensor,
        ), f"Ground truth label must be a sequence of integers or a torch.Tensor, got {type(gt_label)}."
        assert gt_label.ndim == 1, f"Ground truth label must be a 1-dimensional vector, got shape {gt_label.shape}."
        assert (
            len(gt_label) == self.batch_size
        ), f"Ground truth label must have length {self.batch_size}, got length {len(gt_label)}."
        assert not torch.is_floating_point(gt_label), f"Ground truth label must be boolean or integer, got {gt_label}."
        return gt_label.bool()

    def _validate_gt_mask(self, gt_mask: Mask | None) -> Mask | None:
        if gt_mask is None:
            return None
        assert isinstance(gt_mask, torch.Tensor), f"Ground truth mask must be a torch.Tensor, got {type(gt_mask)}."
        assert gt_mask.ndim in {
            2,
            3,
            4,
        }, f"Ground truth mask must have shape [H, W] or [N, H, W] or [N, 1, H, W] got shape {gt_mask.shape}."
        if gt_mask.ndim == 2:
            assert (
                self.batch_size == 1
            ), f"Invalid shape for gt_mask. Got mask shape {gt_mask.shape} for batch size {self.batch_size}."
            gt_mask = gt_mask.unsqueeze(0)
        if gt_mask.ndim == 3:
            assert (
                gt_mask.shape[0] == self.batch_size
            ), f"Invalid shape for gt_mask. Got mask shape {gt_mask.shape} for batch size {self.batch_size}."
        if gt_mask.ndim == 4:
            assert gt_mask.shape[1] == 1, f"Ground truth mask must have 1 channel, got {gt_mask.shape[1]}."
            gt_mask = gt_mask.squeeze(1)
        return Mask(gt_mask, dtype=torch.bool)

    def _validate_mask_path(self, mask_path: Sequence[str] | Sequence[str] | None) -> list[str] | None:
        if mask_path is None:
            return None
        assert isinstance(
            mask_path,
            Sequence,
        ), f"Mask path must be a sequence of paths or strings, got {type(mask_path)}."
        assert (
            len(mask_path) == self.batch_size
        ), f"Invalid length for mask_path. Got length {len(mask_path)} for batch size {self.batch_size}."
        return [str(path) for path in mask_path]

    def _validate_anomaly_map(self, anomaly_map: torch.Tensor | np.ndarray | None) -> torch.Tensor | None:
        if anomaly_map is None:
            return None
        if not isinstance(anomaly_map, torch.Tensor):
            try:
                anomaly_map = torch.tensor(anomaly_map)
            except Exception as e:
                msg = "Failed to convert anomaly_map to a torch.Tensor."
                raise ValueError(msg) from e
        assert anomaly_map.ndim in {
            2,
            3,
            4,
        }, f"Anomaly map must have shape [H, W] or [N, H, W] or [N, 1, H, W], got shape {anomaly_map.shape}."
        if anomaly_map.ndim == 2:
            assert (
                self.batch_size == 1
            ), f"Invalid shape for anomaly_map. Got mask shape {anomaly_map.shape} for batch size {self.batch_size}."
            anomaly_map = anomaly_map.unsqueeze(0)
        if anomaly_map.ndim == 4:
            assert anomaly_map.shape[1] == 1, f"Anomaly map must have 1 channel, got {anomaly_map.shape[1]}."
            anomaly_map = anomaly_map.squeeze(1)
        return Mask(anomaly_map, dtype=torch.float32)

    def _validate_pred_score(self, pred_score: torch.Tensor | None) -> torch.Tensor | None:
        if pred_score is None and self.anomaly_map is not None:
            return torch.amax(self.anomaly_map, dim=(-2, -1))
        return pred_score

    def _validate_pred_mask(self, pred_mask: torch.Tensor) -> torch.Tensor | None:
        return pred_mask

    def _validate_pred_label(self, pred_label: torch.Tensor) -> torch.Tensor | None:
        return pred_label

    def _validate_image_path(self, image_path: list[str]) -> list[str] | None:
        return image_path
