"""Base Depth Dataset."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.transforms.v2 import Transform
from torchvision.tv_tensors import Mask

from anomalib import TaskType
from anomalib.data.base.dataset import AnomalibDataset
from anomalib.data.utils import LabelName, masks_to_boxes, read_depth_image


class AnomalibDepthDataset(AnomalibDataset, ABC):
    """Base depth anomalib dataset class.

    Args:
        task (str): Task type, either 'classification' or 'segmentation'
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
    """

    def __init__(self, task: TaskType, transform: Transform | None = None) -> None:
        super().__init__(task, transform)

        self.transform = transform

    def __getitem__(self, index: int) -> dict[str, str | torch.Tensor]:
        """Return rgb image, depth image and mask.

        Args:
            index (int): Index of the item to be returned.

        Returns:
            dict[str, str | torch.Tensor]: Dictionary containing the image, depth image and mask.
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
                self.transform(image, depth_image) if self.transform else (image, depth_image)
            )
        elif self.task in {TaskType.DETECTION, TaskType.SEGMENTATION}:
            # Only Anomalous (1) images have masks in anomaly datasets
            # Therefore, create empty mask for Normal (0) images.
            mask = (
                Mask(torch.zeros(image.shape[-2:]))
                if label_index == LabelName.NORMAL
                else Mask(to_tensor(Image.open(mask_path)).squeeze())
            )
            item["image"], item["depth_image"], item["mask"] = (
                self.transform(image, depth_image, mask) if self.transform else (image, depth_image, mask)
            )
            item["mask_path"] = mask_path

            if self.task == TaskType.DETECTION:
                # create boxes from masks for detection task
                boxes, _ = masks_to_boxes(item["mask"])
                item["boxes"] = boxes[0]
        else:
            msg = f"Unknown task type: {self.task}"
            raise ValueError(msg)

        return item
