"""Base Depth Dataset.

This module implements the base depth dataset class for anomaly detection tasks that
use RGB-D (RGB + Depth) data.
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from collections.abc import Callable

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.transforms.v2 import Transform
from torchvision.tv_tensors import Mask

from anomalib import TaskType
from anomalib.data.dataclasses import DepthBatch, DepthItem
from anomalib.data.utils import LabelName, read_depth_image

from .image import AnomalibDataset


class AnomalibDepthDataset(AnomalibDataset, ABC):
    """Base depth anomalib dataset class.

    This class extends ``AnomalibDataset`` to handle RGB-D data for anomaly
    detection tasks. It supports both classification and segmentation tasks.

    Args:
        augmentations (Transform, optional): Augmentations that should be applied to the input images.
            Defaults to ``None``.

    Example:
        >>> from anomalib.data.datasets import AnomalibDepthDataset
        >>> dataset = AnomalibDepthDataset(transform=None)
        >>> item = dataset[0]
        >>> item.image.shape
        torch.Size([3, H, W])
        >>> item.depth_map.shape
        torch.Size([1, H, W])
    """

    def __init__(self, augmentations: Transform | None = None) -> None:
        super().__init__(augmentations=augmentations)

        self.augmentations = augmentations

    def __getitem__(self, index: int) -> DepthItem:
        """Get dataset item for the given index.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            DepthItem: Dataset item containing the following fields:
                - image (Tensor): RGB image
                - depth_map (Tensor): Depth map
                - gt_mask (Tensor | None): Ground truth mask for segmentation
                - gt_label (int): Ground truth label (0: normal, 1: anomalous)
                - image_path (str): Path to the RGB image
                - depth_path (str): Path to the depth map
                - mask_path (str | None): Path to the ground truth mask

        Raises:
            ValueError: If the task type is neither classification nor
                segmentation.
        """
        image_path = self.samples.iloc[index].image_path
        mask_path = self.samples.iloc[index].mask_path
        label_index = self.samples.iloc[index].label_index
        depth_path = self.samples.iloc[index].depth_path

        image = to_tensor(Image.open(image_path))
        depth_image = to_tensor(read_depth_image(depth_path))
        item = {"image_path": image_path, "depth_path": depth_path, "label": label_index}

        if self.task == TaskType.CLASSIFICATION:
            item["image"], item["depth_image"] = (
                self.augmentations(image, depth_image) if self.augmentations else (image, depth_image)
            )
        elif self.task == TaskType.SEGMENTATION:
            # Only Anomalous (1) images have masks in anomaly datasets
            # Therefore, create empty mask for Normal (0) images.
            mask = (
                Mask(torch.zeros(image.shape[-2:]))
                if label_index == LabelName.NORMAL
                else Mask(to_tensor(Image.open(mask_path)).squeeze())
            )
            item["image"], item["depth_image"], item["mask"] = (
                self.augmentations(image, depth_image, mask) if self.augmentations else (image, depth_image, mask)
            )
            item["mask_path"] = mask_path

        else:
            msg = f"Unknown task type: {self.task}"
            raise ValueError(msg)

        return DepthItem(
            image=item["image"],
            depth_map=item["depth_image"],
            gt_mask=item.get("mask"),
            gt_label=item["label"],
            image_path=image_path,
            depth_path=depth_path,
            mask_path=item.get("mask_path"),
        )

    @property
    def collate_fn(self) -> Callable:
        """Get the collate function for creating depth batches.

        Returns:
            Callable: Collate function that creates ``DepthBatch`` objects.
        """
        return DepthBatch.collate
