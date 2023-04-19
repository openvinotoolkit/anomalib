"""Anomalib dataset base class."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset

from anomalib.data.task_type import TaskType
from anomalib.data.utils import masks_to_boxes, read_image

_EXPECTED_COLUMNS_CLASSIFICATION = ["image_path", "split"]
_EXPECTED_COLUMNS_SEGMENTATION = _EXPECTED_COLUMNS_CLASSIFICATION + ["mask_path"]
_EXPECTED_COLUMNS_PERTASK = {
    "classification": _EXPECTED_COLUMNS_CLASSIFICATION,
    "segmentation": _EXPECTED_COLUMNS_SEGMENTATION,
    "detection": _EXPECTED_COLUMNS_SEGMENTATION,
}

logger = logging.getLogger(__name__)


class AnomalibDataset(Dataset, ABC):
    """Anomalib dataset.

    Args:
        task (str): Task type, either 'classification' or 'segmentation'
        transform (A.Compose): Albumentations Compose object describing the transforms that are applied to the inputs.
    """

    def __init__(self, task: TaskType, transform: A.Compose) -> None:
        super().__init__()
        self.task = task
        self.transform = transform
        self._samples: DataFrame

    def __len__(self) -> int:
        """Get length of the dataset."""
        return len(self.samples)

    def subsample(self, indices: Sequence[int], inplace: bool = False) -> AnomalibDataset:
        """Subsamples the dataset at the provided indices.

        Args:
            indices (Sequence[int]): Indices at which the dataset is to be subsampled.
            inplace (bool): When true, the subsampling will be performed on the instance itself.
        """
        assert len(set(indices)) == len(indices), "No duplicates allowed in indices."
        dataset = self if inplace else copy.deepcopy(self)
        dataset.samples = self.samples.iloc[indices].reset_index(drop=True)
        return dataset

    @property
    def is_setup(self) -> bool:
        """Checks if setup() been called."""
        return hasattr(self, "_samples")

    @property
    def samples(self) -> DataFrame:
        """Get the samples dataframe."""
        if not self.is_setup:
            raise RuntimeError("Dataset is not setup yet. Call setup() first.")
        return self._samples

    @samples.setter
    def samples(self, samples: DataFrame) -> None:
        """Overwrite the samples with a new dataframe.

        Args:
            samples (DataFrame): DataFrame with new samples.
        """
        # validate the passed samples by checking the
        assert isinstance(samples, DataFrame), f"samples must be a pandas.DataFrame, found {type(samples)}"
        expected_columns = _EXPECTED_COLUMNS_PERTASK[self.task]
        assert all(
            col in samples.columns for col in expected_columns
        ), f"samples must have (at least) columns {expected_columns}, found {samples.columns}"
        assert samples["image_path"].apply(lambda p: Path(p).exists()).all(), "missing file path(s) in samples"

        self._samples = samples.sort_values(by="image_path", ignore_index=True)

    @property
    def has_normal(self) -> bool:
        """Check if the dataset contains any normal samples."""
        return 0 in list(self.samples.label_index)

    @property
    def has_anomalous(self) -> bool:
        """Check if the dataset contains any anomalous samples."""
        return 1 in list(self.samples.label_index)

    def __getitem__(self, index: int) -> dict[str, str | Tensor]:
        """Get dataset item for the index ``index``.

        Args:
            index (int): Index to get the item.

        Returns:
            Union[dict[str, Tensor], dict[str, str | Tensor]]: Dict of image tensor during training.
                Otherwise, Dict containing image path, target path, image tensor, label and transformed bounding box.
        """

        image_path = self._samples.iloc[index].image_path
        mask_path = self._samples.iloc[index].mask_path
        label_index = self._samples.iloc[index].label_index

        image = read_image(image_path)
        item = dict(image_path=image_path, label=label_index)

        if self.task == TaskType.CLASSIFICATION:
            transformed = self.transform(image=image)
            item["image"] = transformed["image"]
        elif self.task in (TaskType.DETECTION, TaskType.SEGMENTATION):
            # Only Anomalous (1) images have masks in anomaly datasets
            # Therefore, create empty mask for Normal (0) images.

            if label_index == 0:
                mask = np.zeros(shape=image.shape[:2])
            else:
                mask = cv2.imread(mask_path, flags=0) / 255.0

            transformed = self.transform(image=image, mask=mask)

            item["image"] = transformed["image"]
            item["mask_path"] = mask_path
            item["mask"] = transformed["mask"]

            if self.task == TaskType.DETECTION:
                # create boxes from masks for detection task
                boxes, _ = masks_to_boxes(item["mask"])
                item["boxes"] = boxes[0]
        else:
            raise ValueError(f"Unknown task type: {self.task}")

        return item

    def __add__(self, other_dataset: AnomalibDataset) -> AnomalibDataset:
        """Concatenate this dataset with another dataset."""
        assert isinstance(other_dataset, self.__class__), "Cannot concatenate datasets that are not of the same type."
        assert self.is_setup, "Cannot concatenate uninitialized datasets. Call setup first."
        assert other_dataset.is_setup, "Cannot concatenate uninitialized datasets. Call setup first."
        dataset = copy.deepcopy(self)
        dataset.samples = pd.concat([self.samples, other_dataset.samples], ignore_index=True)
        return dataset

    def setup(self) -> None:
        """Load data/metadata into memory."""
        if not self.is_setup:
            self._setup()
        assert self.is_setup, "setup() should set self._samples"

    @abstractmethod
    def _setup(self) -> DataFrame:
        """Set up the data module.

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
        | 0 | path/to/image.png | anomalous | 1           | path/to/mask.png | train |
        |---|-------------------|-----------|-------------|------------------|-------|
        """
        raise NotImplementedError
