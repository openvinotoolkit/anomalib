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

from anomalib.data.generic import BatchIterateMixin, _DepthInputFields
from anomalib.data.numpy.image import NumpyImageItem
from anomalib.data.torch.base import Batch, DatasetItem, ToNumpyMixin


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

    def _validate_image(self, image: Image) -> Image:
        return image

    def _validate_gt_label(self, gt_label: torch.Tensor) -> torch.Tensor:
        return gt_label

    def _validate_gt_mask(self, gt_mask: Mask) -> Mask:
        return gt_mask

    def _validate_mask_path(self, mask_path: str) -> str:
        return mask_path

    def _validate_anomaly_map(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        return anomaly_map

    def _validate_pred_score(self, pred_score: torch.Tensor) -> torch.Tensor:
        return pred_score

    def _validate_pred_mask(self, pred_mask: torch.Tensor) -> torch.Tensor:
        return pred_mask

    def _validate_pred_label(self, pred_label: torch.Tensor) -> torch.Tensor:
        return pred_label

    def _validate_image_path(self, image_path: str) -> str:
        return image_path

    def _validate_depth_map(self, depth_map: torch.Tensor) -> torch.Tensor:
        return depth_map

    def _validate_depth_path(self, depth_path: str) -> str:
        return depth_path


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

    def _validate_image(self, image: Image) -> Image:
        return image

    def _validate_gt_label(self, gt_label: torch.Tensor) -> torch.Tensor:
        return gt_label

    def _validate_gt_mask(self, gt_mask: Mask) -> Mask:
        return gt_mask

    def _validate_mask_path(self, mask_path: list[str]) -> list[str]:
        return mask_path

    def _validate_anomaly_map(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        return anomaly_map

    def _validate_pred_score(self, pred_score: torch.Tensor) -> torch.Tensor:
        return pred_score

    def _validate_pred_mask(self, pred_mask: torch.Tensor) -> torch.Tensor:
        return pred_mask

    def _validate_pred_label(self, pred_label: torch.Tensor) -> torch.Tensor:
        return pred_label

    def _validate_image_path(self, image_path: list[str]) -> list[str]:
        return image_path

    def _validate_depth_map(self, depth_map: torch.Tensor) -> torch.Tensor:
        return depth_map

    def _validate_depth_path(self, depth_path: list[str]) -> list[str]:
        return depth_path
