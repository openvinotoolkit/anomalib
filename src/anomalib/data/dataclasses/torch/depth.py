"""Torch-based dataclasses for depth data in Anomalib.

This module provides PyTorch-based implementations of the generic dataclasses
used in Anomalib for depth data. These classes are designed to work with PyTorch
tensors for efficient data handling and processing in anomaly detection tasks.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from torchvision.tv_tensors import Image

from anomalib.data.dataclasses.generic import BatchIterateMixin, _DepthInputFields
from anomalib.data.dataclasses.numpy.image import NumpyImageItem
from anomalib.data.dataclasses.torch.base import Batch, DatasetItem, ToNumpyMixin
from anomalib.data.validators.torch.depth import DepthBatchValidator, DepthValidator


@dataclass
class DepthItem(
    ToNumpyMixin[NumpyImageItem],
    DepthValidator,
    _DepthInputFields[torch.Tensor, str],
    DatasetItem[Image],
):
    """Dataclass for individual depth items in Anomalib datasets using PyTorch.

    This class represents a single depth item in Anomalib datasets using PyTorch
    tensors. It combines the functionality of ``ToNumpyMixin``,
    ``_DepthInputFields``, and ``DatasetItem`` to handle depth data, including
    depth maps, labels, and metadata.

    Args:
        image (torch.Tensor): Image tensor of shape ``(C, H, W)``.
        gt_label (torch.Tensor): Ground truth label tensor.
        depth_map (torch.Tensor): Depth map tensor of shape ``(H, W)``.
        image_path (str): Path to the source image file.
        depth_path (str): Path to the depth map file.

    Examples:
        >>> import torch
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


@dataclass
class DepthBatch(
    BatchIterateMixin[DepthItem],
    DepthBatchValidator,
    _DepthInputFields[torch.Tensor, list[str]],
    Batch[Image],
):
    """Dataclass for batches of depth items in Anomalib datasets using PyTorch.

    This class represents a batch of depth items in Anomalib datasets using
    PyTorch tensors. It combines the functionality of ``BatchIterateMixin``,
    ``_DepthInputFields``, and ``Batch`` to handle batches of depth data,
    including depth maps, labels, and metadata.

    Args:
        image (torch.Tensor): Batch of images of shape ``(B, C, H, W)``.
        gt_label (torch.Tensor): Batch of ground truth labels of shape ``(B,)``.
        depth_map (torch.Tensor): Batch of depth maps of shape ``(B, H, W)``.
        image_path (list[str]): List of paths to the source image files.
        depth_path (list[str]): List of paths to the depth map files.

    Examples:
        >>> import torch
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
