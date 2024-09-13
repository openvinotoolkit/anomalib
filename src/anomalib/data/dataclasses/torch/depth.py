"""Torch-based dataclasses for depth data in Anomalib.

This module provides PyTorch-based implementations of the generic dataclasses
used in Anomalib for depth data. These classes are designed to work with PyTorch
tensors for efficient data handling and processing in anomaly detection tasks.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from torchvision.tv_tensors import Image, Mask

from anomalib.data.dataclasses.generic import BatchIterateMixin, _DepthInputFields
from anomalib.data.dataclasses.numpy.image import NumpyImageItem
from anomalib.data.dataclasses.torch.base import Batch, DatasetItem, ToNumpyMixin
from anomalib.data.validators.torch.depth import DepthBatchValidator, DepthValidator


@dataclass
class DepthItem(
    ToNumpyMixin[NumpyImageItem],
    _DepthInputFields[torch.Tensor, str],
    DatasetItem[Image],
):
    """Dataclass for individual depth items in Anomalib datasets using PyTorch tensors.

    This class represents a single depth item in Anomalib datasets using PyTorch tensors.
    It combines the functionality of ToNumpyMixin, _DepthInputFields, and DatasetItem
    to handle depth data, including depth maps, labels, and metadata.

    Examples:
        >>> item = DepthItem(
        ...     image=torch.rand(3, 224, 224),
        ...     gt_label=torch.tensor(1),
        ...     depth_map=torch.rand(224, 224),
        ...     image_path="path/to/image.jpg",
        ...     depth_path="path/to/depth.png"
        ... )

        >>> print(item.image.shape, item.depth_map.shape)
        torch.Size([3, 224, 224]) torch.Size([224, 224])
    """

    numpy_class = NumpyImageItem

    @staticmethod
    def validate_image(image: Image) -> Image:
        """Validate the image."""
        return DepthValidator.validate_image(image)

    @staticmethod
    def validate_gt_label(gt_label: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the ground truth label."""
        return DepthValidator.validate_gt_label(gt_label)

    @staticmethod
    def validate_gt_mask(gt_mask: Mask | None) -> Mask | None:
        """Validate the ground truth mask."""
        return DepthValidator.validate_gt_mask(gt_mask)

    @staticmethod
    def validate_mask_path(mask_path: str | None) -> str | None:
        """Validate the mask path."""
        return DepthValidator.validate_mask_path(mask_path)

    @staticmethod
    def validate_anomaly_map(anomaly_map: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the anomaly map."""
        return DepthValidator.validate_anomaly_map(anomaly_map)

    @staticmethod
    def validate_pred_score(pred_score: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the prediction score."""
        return DepthValidator.validate_pred_score(pred_score)

    @staticmethod
    def validate_pred_mask(pred_mask: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the prediction mask."""
        return DepthValidator.validate_pred_mask(pred_mask)

    @staticmethod
    def validate_pred_label(pred_label: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the prediction label."""
        return DepthValidator.validate_pred_label(pred_label)

    @staticmethod
    def validate_image_path(image_path: str | None) -> str | None:
        """Validate the image path."""
        return DepthValidator.validate_image_path(image_path)

    @staticmethod
    def validate_depth_map(depth_map: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the depth map."""
        return DepthValidator.validate_depth_map(depth_map)

    @staticmethod
    def validate_depth_path(depth_path: str | None) -> str | None:
        """Validate the depth path."""
        return DepthValidator.validate_depth_path(depth_path)


@dataclass
class DepthBatch(
    BatchIterateMixin[DepthItem],
    _DepthInputFields[torch.Tensor, list[str]],
    Batch[Image],
):
    """Dataclass for batches of depth items in Anomalib datasets using PyTorch tensors.

    This class represents a batch of depth items in Anomalib datasets using PyTorch tensors.
    It combines the functionality of BatchIterateMixin, _DepthInputFields, and Batch
    to handle batches of depth data, including depth maps, labels, and metadata.

    Examples:
        >>> batch = DepthBatch(
        ...     image=torch.rand(32, 3, 224, 224),
        ...     gt_label=torch.randint(0, 2, (32,)),
        ...     depth_map=torch.rand(32, 224, 224),
        ...     image_path=["path/to/image_{}.jpg".format(i) for i in range(32)],
        ...     depth_path=["path/to/depth_{}.png".format(i) for i in range(32)]
        ... )

        >>> print(batch.image.shape, batch.depth_map.shape)
        torch.Size([32, 3, 224, 224]) torch.Size([32, 224, 224])

        >>> for item in batch:
        ...     print(item.image.shape, item.depth_map.shape)
        torch.Size([3, 224, 224]) torch.Size([224, 224])
    """

    item_class = DepthItem

    @staticmethod
    def validate_image(image: Image) -> Image:
        """Validate the image."""
        return DepthBatchValidator.validate_image(image)

    def validate_gt_label(self, gt_label: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the ground truth label."""
        return DepthBatchValidator.validate_gt_label(gt_label, self.batch_size)

    def validate_gt_mask(self, gt_mask: Mask | None) -> Mask | None:
        """Validate the ground truth mask."""
        return DepthBatchValidator.validate_gt_mask(gt_mask, self.batch_size)

    def validate_mask_path(self, mask_path: list[str] | None) -> list[str] | None:
        """Validate the mask path."""
        return DepthBatchValidator.validate_mask_path(mask_path, self.batch_size)

    def validate_anomaly_map(self, anomaly_map: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the anomaly map."""
        return DepthBatchValidator.validate_anomaly_map(anomaly_map, self.batch_size)

    def validate_pred_score(self, pred_score: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the prediction score."""
        return DepthBatchValidator.validate_pred_score(pred_score, self.anomaly_map)

    def validate_pred_mask(self, pred_mask: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the prediction mask."""
        return DepthBatchValidator.validate_pred_mask(pred_mask, self.batch_size)

    @staticmethod
    def validate_pred_label(pred_label: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the prediction label."""
        return DepthBatchValidator.validate_pred_label(pred_label)

    @staticmethod
    def validate_image_path(image_path: list[str] | None) -> list[str] | None:
        """Validate the image path."""
        return DepthBatchValidator.validate_image_path(image_path)

    def validate_depth_map(self, depth_map: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the depth map."""
        return DepthBatchValidator.validate_depth_map(depth_map, self.batch_size)

    @staticmethod
    def validate_depth_path(depth_path: list[str] | None) -> list[str] | None:
        """Validate the depth path."""
        return DepthBatchValidator.validate_depth_path(depth_path)
