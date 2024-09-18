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
    def _validate_image(image: Image) -> Image:
        return image

    @staticmethod
    def _validate_gt_label(gt_label: torch.Tensor) -> torch.Tensor:
        return gt_label

    @staticmethod
    def _validate_gt_mask(gt_mask: Mask) -> Mask:
        return gt_mask

    @staticmethod
    def _validate_mask_path(mask_path: str) -> str:
        return mask_path

    @staticmethod
    def _validate_anomaly_map(anomaly_map: torch.Tensor) -> torch.Tensor:
        return anomaly_map

    @staticmethod
    def _validate_pred_score(pred_score: torch.Tensor) -> torch.Tensor:
        return pred_score

    @staticmethod
    def _validate_pred_mask(pred_mask: torch.Tensor) -> torch.Tensor:
        return pred_mask

    @staticmethod
    def _validate_pred_label(pred_label: torch.Tensor) -> torch.Tensor:
        return pred_label

    @staticmethod
    def _validate_image_path(image_path: str) -> str:
        return image_path

    @staticmethod
    def _validate_depth_map(depth_map: torch.Tensor) -> torch.Tensor:
        return depth_map

    @staticmethod
    def _validate_depth_path(depth_path: str) -> str:
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

    @staticmethod
    def _validate_image(image: Image) -> Image:
        return image

    @staticmethod
    def _validate_gt_label(gt_label: torch.Tensor) -> torch.Tensor:
        return gt_label

    @staticmethod
    def _validate_gt_mask(gt_mask: Mask) -> Mask:
        return gt_mask

    @staticmethod
    def _validate_mask_path(mask_path: list[str]) -> list[str]:
        return mask_path

    @staticmethod
    def _validate_anomaly_map(anomaly_map: torch.Tensor) -> torch.Tensor:
        return anomaly_map

    @staticmethod
    def _validate_pred_score(pred_score: torch.Tensor) -> torch.Tensor:
        return pred_score

    @staticmethod
    def _validate_pred_mask(pred_mask: torch.Tensor) -> torch.Tensor:
        return pred_mask

    @staticmethod
    def _validate_pred_label(pred_label: torch.Tensor) -> torch.Tensor:
        return pred_label

    @staticmethod
    def _validate_image_path(image_path: list[str]) -> list[str]:
        return image_path

    @staticmethod
    def _validate_depth_map(depth_map: torch.Tensor) -> torch.Tensor:
        return depth_map

    @staticmethod
    def _validate_depth_path(depth_path: list[str]) -> list[str]:
        return depth_path
