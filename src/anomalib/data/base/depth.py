"""Base Depth Dataset."""


from abc import ABC

import albumentations as A  # noqa: N812
import cv2
import numpy as np
import torch

from anomalib.data.base.dataset import AnomalibDataset
from anomalib.data.utils import masks_to_boxes, read_depth_image, read_image
from anomalib.utils.types import TaskType


class AnomalibDepthDataset(AnomalibDataset, ABC):
    """Base depth anomalib dataset class.

    Args:
        task (str): Task type, either 'classification' or 'segmentation'
        transform (A.Compose): Albumentations Compose object describing the transforms that are applied to the inputs.
    """

    def __init__(self, task: TaskType, transform: A.Compose) -> None:
        super().__init__(task, transform)

        self.transform = transform

    def __getitem__(self, index: int) -> dict[str, str | torch.Tensor]:
        """Return rgb image, depth image and mask.

        Args:
            index (int): Index of the item to be returned.

        Returns:
            dict[str, str | torch.Tensor]: Dictionary containing the image, depth image and mask.
        """
        image_path = self._samples.iloc[index].image_path
        mask_path = self._samples.iloc[index].mask_path
        label_index = self._samples.iloc[index].label_index
        depth_path = self._samples.iloc[index].depth_path

        image = read_image(image_path)
        depth_image = read_depth_image(depth_path)
        item = {"image_path": image_path, "depth_path": depth_path, "label": label_index}

        if self.task == TaskType.CLASSIFICATION:
            transformed = self.transform(image=image, depth_image=depth_image)
            item["image"] = transformed["image"]
            item["depth_image"] = transformed["depth_image"]
        elif self.task in (TaskType.DETECTION, TaskType.SEGMENTATION):
            # Only Anomalous (1) images have masks in anomaly datasets
            # Therefore, create empty mask for Normal (0) images.
            mask = np.zeros(shape=image.shape[:2]) if label_index == 0 else cv2.imread(mask_path, flags=0) / 255.0

            transformed = self.transform(image=image, depth_image=depth_image, mask=mask)

            item["image"] = transformed["image"]
            item["depth_image"] = transformed["depth_image"]
            item["mask_path"] = mask_path
            item["mask"] = transformed["mask"]

            if self.task == TaskType.DETECTION:
                # create boxes from masks for detection task
                boxes, _ = masks_to_boxes(item["mask"])
                item["boxes"] = boxes[0]
        else:
            msg = f"Unknown task type: {self.task}"
            raise ValueError(msg)

        return item
