"""Anomalib dataset and datamodule base classes."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, Union

import cv2
import numpy as np
import pandas as pd
from pandas import DataFrame
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from anomalib.data.utils import read_image
from anomalib.pre_processing import PreProcessor

logger = logging.getLogger(__name__)


class Split(str, Enum):
    FULL = "full"
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class ValSplitMode(str, Enum):
    SAME_AS_TEST = "same_as_test"
    FROM_TEST = "from_test"


class AnomalibDataset(Dataset):
    """Anomalib dataset."""

    def __init__(self, task: str, pre_process: PreProcessor, samples: Optional[DataFrame] = None):
        super().__init__()
        self.task = task
        self.pre_process = pre_process
        self._samples = samples

    def __len__(self) -> int:
        """Get length of the dataset."""
        assert isinstance(self._samples, DataFrame)
        return len(self._samples)

    def subsample(self, indices):
        return AnomalibDataset(task=self.task, pre_process=self.pre_process, samples=self.samples.iloc[indices])

    @property
    def is_setup(self) -> bool:
        """Has setup() been called?"""
        return isinstance(self._samples, DataFrame)

    @property
    def samples(self) -> DataFrame:
        """TODO"""
        if not self.is_setup:
            raise RuntimeError("Dataset is not setup yet. Call setup() first.")
        return self._samples

    @property
    def has_normal(self) -> bool:
        return 0 in list(self.samples.label_index)

    @property
    def has_anomalous(self) -> bool:
        return 1 in list(self.samples.label_index)

    def __getitem__(self, index: int) -> Dict[str, Union[str, Tensor]]:
        """Get dataset item for the index ``index``.

        Args:
            index (int): Index to get the item.

        Returns:
            Union[Dict[str, Tensor], Dict[str, Union[str, Tensor]]]: Dict of image tensor during training.
                Otherwise, Dict containing image path, target path, image tensor, label and transformed bounding box.
        """
        assert isinstance(self._samples, DataFrame)

        image_path = self._samples.iloc[index].image_path
        image = read_image(image_path)
        label_index = self._samples.iloc[index].label_index

        item = dict(image_path=image_path, label=label_index)

        if self.task == "classification":
            pre_processed = self.pre_process(image=image)
        elif self.task == "segmentation":
            mask_path = self._samples.iloc[index].mask_path

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

    def __add__(self, other_dataset: AnomalibDataset):
        assert self.is_setup and other_dataset.is_setup, "Cannot concatenate uninitialized datasets. Call setup first."
        samples = pd.concat([self.samples, other_dataset.samples], ignore_index=True)
        return AnomalibDataset(self.task, self.pre_process, samples)

    def setup(self) -> None:
        """Load data/metadata into memory"""
        if not self.is_setup:
            self._setup()
        assert self.is_setup, "setup() should set self._samples"

    def _setup(self) -> DataFrame:
        """previous _create_samples()
        This method should return a dataframe that contains the information needed by the dataloader to load each of
        the dataset items into memory.
        The dataframe must at least contain the following columns:
            split: the subset to which the dataset item is assigned.
            image_path: path to file system location where the image is stored.
            label_index: index of the anomaly label, typically 0 for "normal" and 1 for "anomalous".
            mask_path (if task == "segmentation"): path to the ground truth masks (for the anomalous images only).
        Example:
        |---|-------------------|-----------|-------------|------------------|-------|
        |   | image_path        | label     | label_index | mask_path        | split |
        |---|-------------------|-----------|-------------|------------------|-------|
        | 0 | path/to/image.png | anomalous | 0           | path/to/mask.png | train |
        |---|-------------------|-----------|-------------|------------------|-------|
        """
        pass


class AnomalibDataModule(LightningDataModule, ABC):
    """Base Anomalib data module."""

    def __init__(
        self,
        train_batch_size: int,
        test_batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        self.train_data: Optional[AnomalibDataset] = None
        self.val_data: Optional[AnomalibDataset] = None
        self.test_data: Optional[AnomalibDataset] = None

        self._samples: Optional[DataFrame] = None

    def setup(self, stage: Optional[str] = None):
        """Setup train, validation and test data.

        Args:
          stage: Optional[str]:  Train/Val/Test stages. (Default value = None)
        """
        if not self.is_setup:
            self._setup(stage)
        assert self.is_setup

    @abstractmethod
    def _setup(self, _stage: Optional[str] = None) -> None:
        pass

    @property
    def is_setup(self):
        if self.train_data is None or self.val_data is None or self.test_data is None:
            return False
        return self.train_data.is_setup and self.val_data.is_setup and self.test_data.is_setup

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Get train dataloader."""
        return DataLoader(self.train_data, shuffle=True, batch_size=self.train_batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Get validation dataloader."""
        return DataLoader(self.val_data, shuffle=False, batch_size=self.test_batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Get test dataloader."""
        return DataLoader(self.test_data, shuffle=False, batch_size=self.test_batch_size, num_workers=self.num_workers)
