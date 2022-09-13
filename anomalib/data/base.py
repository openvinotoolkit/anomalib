"""Anomalib dataset and datamodule base classes."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from typing import Dict, Optional, Union

import cv2
import numpy as np
from pandas import DataFrame
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import Dataset

from anomalib.data.utils import read_image
from anomalib.pre_processing import PreProcessor


class AnomalibDataset(Dataset):
    """Anomalib dataset."""

    def __init__(self, samples: DataFrame, task: str, split: str, pre_process: PreProcessor):
        super().__init__()
        self.samples = samples
        self.task = task
        self.split = split
        self.pre_process = pre_process

    def contains_anomalous_images(self):
        """Check if the dataset contains any anomalous images."""
        return "anomalous" in list(self.samples.label)

    def __len__(self) -> int:
        """Get length of the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Union[str, Tensor]]:
        """Get dataset item for the index ``index``.

        Args:
            index (int): Index to get the item.

        Returns:
            Union[Dict[str, Tensor], Dict[str, Union[str, Tensor]]]: Dict of image tensor during training.
                Otherwise, Dict containing image path, target path, image tensor, label and transformed bounding box.
        """
        image_path = self.samples.image_path[index]
        image = read_image(image_path)

        pre_processed = self.pre_process(image=image)
        item = {"image": pre_processed["image"]}

        if self.split in ["val", "test"]:
            label_index = self.samples.label_index[index]

            item["image_path"] = image_path
            item["label"] = label_index

            if self.task == "segmentation":
                mask_path = self.samples.mask_path[index]

                # Only Anomalous (1) images has masks in MVTec AD dataset.
                # Therefore, create empty mask for Normal (0) images.
                if label_index == 0:
                    mask = np.zeros(shape=image.shape[:2])
                else:
                    mask = cv2.imread(mask_path, flags=0) / 255.0

                pre_processed = self.pre_process(image=image, mask=mask)

                item["mask_path"] = mask_path
                item["image"] = pre_processed["image"]
                item["mask"] = pre_processed["mask"]

        return item


class AnomalibDataModule(LightningDataModule, ABC):
    """Base Anomalib data module."""

    def __init__(self):
        super().__init__()
        self.train_data: Optional[AnomalibDataset] = None
        self.val_data: Optional[AnomalibDataset] = None
        self.test_data: Optional[AnomalibDataset] = None
