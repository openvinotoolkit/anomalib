"""Anomalib dataset and datamodule base classes."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
from pandas import DataFrame
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import Dataset

from anomalib.data.utils import read_image
from anomalib.pre_processing import PreProcessor

logger = logging.getLogger(__name__)


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
        return 1 in list(self.samples.label_index)

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
        image_path = self.samples.iloc[index].image_path
        image = read_image(image_path)
        label_index = self.samples.iloc[index].label_index

        item = dict(image_path=image_path, label=label_index)

        if self.task == "classification":
            pre_processed = self.pre_process(image=image)
        elif self.task == "segmentation":
            mask_path = self.samples.iloc[index].mask_path

            # Only Anomalous (1) images have masks in anomaly datasets
            # Therefore, create empty mask for Normal (0) images.
            if label_index == 0:
                mask = np.zeros(shape=image.shape[:2])
            else:
                mask = cv2.imread(mask_path, flags=0) / 255.0

            pre_processed = self.pre_process(image=image, mask=mask)

            item["mask_path"] = mask_path
            item["mask"] = pre_processed["mask"]
        else:
            raise ValueError(f"Unknown task type: {self.task}")
        item["image"] = pre_processed["image"]

        return item


class AnomalibDataModule(LightningDataModule, ABC):
    """Base Anomalib data module."""

    def __init__(
        self,
        task: str,
        transform_config_train: Optional[Union[str, A.Compose]] = None,
        transform_config_val: Optional[Union[str, A.Compose]] = None,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        create_validation_set: bool = False,
    ):
        super().__init__()
        self.task = task
        self.create_validation_set = create_validation_set

        if transform_config_train is not None and transform_config_val is None:
            transform_config_val = transform_config_train
        self.pre_process_train = PreProcessor(config=transform_config_train, image_size=image_size)
        self.pre_process_val = PreProcessor(config=transform_config_val, image_size=image_size)

        self.train_data: Optional[AnomalibDataset] = None
        self.val_data: Optional[AnomalibDataset] = None
        self.test_data: Optional[AnomalibDataset] = None

    @abstractmethod
    def _create_samples(self) -> DataFrame:
        """This method should be implemented in the subclass.

        This method should return a dataframe that contains the information needed by the dataloader to load each of
        the dataset items into memory. The dataframe must at least contain the following columns:
        split - The subset to which the dataset item is assigned.
        image_path - Path to file system location where the image is stored.
        label_index - Index of the anomaly label, typically 0 for "normal" and 1 for "anomalous".

        Additionally, when the task type is segmentation, the dataframe must have the mask_path column, which contains
        the path the ground truth masks (for the anomalous images only).

        Example of a dataframe returned by calling this method from a concrete class:
        |---|-------------------|-----------|-------------|------------------|-------|
        |   | image_path        | label     | label_index | mask_path        | split |
        |---|-------------------|-----------|-------------|------------------|-------|
        | 0 | path/to/image.png | anomalous | 0           | path/to/mask.png | train |
        |---|-------------------|-----------|-------------|------------------|-------|
        """
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup train, validation and test data.

        Args:
          stage: Optional[str]:  Train/Val/Test stages. (Default value = None)

        """
        samples = self._create_samples()

        logger.info("Setting up train, validation, test and prediction datasets.")
        if stage in (None, "fit"):
            train_samples = samples[samples.split == "train"]
            train_samples = train_samples.reset_index(drop=True)
            self.train_data = AnomalibDataset(
                samples=train_samples,
                split="train",
                task=self.task,
                pre_process=self.pre_process_train,
            )

        if self.create_validation_set:
            val_samples = samples[samples.split == "val"]
            val_samples = val_samples.reset_index(drop=True)
            self.val_data = AnomalibDataset(
                samples=val_samples,
                split="val",
                task=self.task,
                pre_process=self.pre_process_val,
            )

        test_samples = samples[samples.split == "test"]
        test_samples = test_samples.reset_index(drop=True)
        self.test_data = AnomalibDataset(
            samples=test_samples,
            split="test",
            task=self.task,
            pre_process=self.pre_process_val,
        )
