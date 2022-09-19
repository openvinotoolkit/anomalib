"""Anomalib dataset and datamodule base classes."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import albumentations as A
import numpy as np
import pandas as pd
from pandas import DataFrame
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from anomalib.data.utils import read_image, read_mask
from anomalib.pre_processing import PreProcessor

logger = logging.getLogger(__name__)

TASK_CLASSIFICATION = "classification"
TASK_SEGMENTATION = "segmentation"
TASKS = (TASK_CLASSIFICATION, TASK_SEGMENTATION)

SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"
SPLITS = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)

LABEL_NORMAL = 0
LABEL_ANOMALOUS = 1

_SAMPLES_MANDATORY_COLUMNS_CLASSIFICATION = ["image_path", "label"]
_SAMPLES_MANDATORY_COLUMNS_SEGMENTATION = _SAMPLES_MANDATORY_COLUMNS_CLASSIFICATION + ["mask_path"]


class AnomalibDataset(Dataset):
    """Anomalib dataset."""

    @staticmethod
    @abstractmethod
    def _create_samples(root: Path, split: str) -> DataFrame:
        """This method should be implemented in the subclass.

        This method should return a dataframe that contains the information needed by __getitem__ to load each of
        the dataset items into memory. The dataframe must at least contain the following columns:
            image_path - Path to file system location where the image is stored.
            label - Index of the anomaly label, typically 0 for "normal" and 1 for "anomalous".

        Additionally, when the task type is segmentation, the dataframe must have the mask_path column, which contains
        the path the ground truth masks (for the anomalous images only).

        Example of a dataframe returned by calling this method from a concrete class:
        |---|-------------------|-----------|-------------|------------------|
        |   | image_path        | label_str | label       | mask_path        |
        |---|-------------------|-----------|-------------|------------------|
        | 0 | path/to/image.png | anomalous | 0           | path/to/mask.png |
        |---|-------------------|-----------|-------------|------------------|
        """
        pass

    def __init__(self, root: Path, task: str, split: str, pre_process: PreProcessor):
        super().__init__()

        assert split in SPLITS, f"Unknown split: {split}"
        assert task in TASKS, f"Unknown task: {task}"

        self.root = root
        self.task = task
        self.split = split
        self.pre_process = pre_process
        self.samples: DataFrame = self._create_samples(self.root, self.split)

        if self.task == TASK_CLASSIFICATION:
            mandatory_columns = _SAMPLES_MANDATORY_COLUMNS_CLASSIFICATION

        elif self.task == TASK_SEGMENTATION:
            mandatory_columns = _SAMPLES_MANDATORY_COLUMNS_SEGMENTATION

        for column in mandatory_columns:
            if column not in self.samples.columns:
                raise ValueError(f"Column {column} is missing in the samples dataframe.")

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
        label = self.samples.iloc[index].label

        item = dict(image_path=image_path, label=label)

        if self.task == "classification":
            pre_processed = self.pre_process(image=image)

        elif self.task == "segmentation":
            mask_path = self.samples.iloc[index].mask_path

            # Only Anomalous (1) images have masks in anomaly datasets
            # Therefore, create empty mask for Normal (0) images.
            if label == LABEL_NORMAL:
                mask = np.zeros(shape=image.shape[:2])

            else:
                mask = read_mask(mask_path)

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
        root: Union[str, Path],
        task: str,
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
        num_workers: int,
        transform_config_train: Optional[Union[str, A.Compose]] = None,
        transform_config_val: Optional[Union[str, A.Compose]] = None,
        transform_config_test: Optional[Union[str, A.Compose]] = None,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        create_validation_set: bool = False,
    ):
        super().__init__()

        self.root = Path(root) if not isinstance(root, Path) else root
        self.task = task

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

        self.num_workers = num_workers
        self.create_validation_set = create_validation_set

        self.pre_process_train = PreProcessor(config=transform_config_train, image_size=image_size)
        self.pre_process_val = PreProcessor(config=transform_config_val, image_size=image_size)
        self.pre_process_test = PreProcessor(config=transform_config_test, image_size=image_size)

        self.train_data: Optional[AnomalibDataset] = None
        self.val_data: Optional[AnomalibDataset] = None
        self.test_data: Optional[AnomalibDataset] = None

    def get_samples(self, split: Optional[str] = None) -> DataFrame:
        """Retrieve the samples of the full dataset or one of the splits (train, val, test).

        Args:
            split: (str): The split for which we want to retrieve the samples ("train", "val" or "test"). When
                left empty, all samples will be returned.

        Returns:
            DataFrame: A dataframe containing the samples of the split or full dataset.
        """
        assert split in SPLITS, f"Unknown split: {split}"

        all_samples = []

        if split in (None, SPLIT_TRAIN):

            if self.train_data is None:
                raise RuntimeError("Train data not initialized. Call setup('fit') first.")

            samples_ = self.train_data.samples.copy()
            samples_["split"] = SPLIT_TRAIN
            all_samples.append(samples_)

        if split in (None, SPLIT_VAL):

            if self.val_data is None:
                raise RuntimeError("Val data not initialized. Call setup('fit' or 'validate') first.")

            samples_ = self.val_data.samples.copy()
            samples_["split"] = SPLIT_VAL
            all_samples.append(samples_)

        if split in (None, SPLIT_TEST):

            if self.test_data is None:
                raise RuntimeError("Test data not initialized. Call setup('test') first.")

            samples_ = self.test_data.samples.copy()
            samples_["split"] = SPLIT_TEST
            all_samples.append(samples_)

        return pd.concat(all_samples).reset_index(drop=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup train, validation and test data.

        If `stage` is `None`, all splits are setup.

        Args:
          stage: Optional[str]:  fit/validate/test stages. (Default value = None, all are set up)
        """

        # setup() can also be called (from pytorch lightning) with stage="predict" and stage="tune"
        # since these two cases are not dealt with in this method, it should break the execution
        # because otherwise the code will continue silently and the user will not be aware
        assert stage in (None, TrainerFn.FITTING, TrainerFn.VALIDATING, TrainerFn.TESTING), f"Unknown stage: {stage}"

        if stage in (None, TrainerFn.FITTING):
            logger.info("Setting up train dataset.")
            self.train_data = AnomalibDataset(
                root=self.root,
                split=SPLIT_TRAIN,
                task=self.task,
                pre_process=self.pre_process_train,
            )

        if stage in (None, TrainerFn.FITTING, TrainerFn.VALIDATING):
            logger.info("Setting up validation dataset.")

            if self.create_validation_set:
                split_ = SPLIT_VAL

            else:
                warnings.warn("Validation split is not availabe. Test split will be used for validation.")
                split_ = SPLIT_TEST

            self.val_data = AnomalibDataset(
                root=self.root,
                split=split_,
                task=self.task,
                pre_process=self.pre_process_val,
            )

        if stage in (None, TrainerFn.TESTING):
            logger.info("Setting up test dataset.")
            self.test_data = AnomalibDataset(
                root=self.root,
                split=SPLIT_TEST,
                task=self.task,
                pre_process=self.pre_process_test,
            )

    def contains_anomalous_images(self, split: Optional[str] = None) -> bool:
        """Check if the dataset or the specified subset contains any anomalous images.

        Args:
            split (str): the subset of interest ("train", "val" or "test"). When left empty, the full dataset will be
                checked.

        Returns:
            bool: Boolean indicating if any anomalous images have been assigned to the dataset or subset.
        """
        samples = self.get_samples(split=split)
        return LABEL_ANOMALOUS in list(samples.label)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Get train dataloader."""
        return DataLoader(self.train_data, shuffle=True, batch_size=self.train_batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Get validation dataloader."""
        return DataLoader(self.val_data, shuffle=False, batch_size=self.val_batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Get test dataloader."""
        return DataLoader(self.test_data, shuffle=False, batch_size=self.test_batch_size, num_workers=self.num_workers)
