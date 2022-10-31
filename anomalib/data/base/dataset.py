"""Anomalib dataset base class."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Sequence, Union

import cv2
import numpy as np
import pandas as pd
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset

from anomalib.data.utils import read_image
from anomalib.pre_processing import PreProcessor

_EXPECTED_COLS_CLASSIFICATION = ["image_path", "label", "label_index", "split"]
_EXPECTED_COLS_SEGMENTATION = _EXPECTED_COLS_CLASSIFICATION + ["mask_path"]
_EXPECTED_COLS_PERTASK = {
    "classification": _EXPECTED_COLS_CLASSIFICATION,
    "segmentation": _EXPECTED_COLS_SEGMENTATION,
}

logger = logging.getLogger(__name__)


class AnomalibDataset(Dataset, ABC):
    """Anomalib dataset."""

    def __init__(self, task: str, pre_process: PreProcessor):
        super().__init__()
        self.task = task
        self.pre_process = pre_process
        self._samples: DataFrame = None

    def __len__(self) -> int:
        """Get length of the dataset."""
        return len(self.samples)

    def subsample(self, indices: Sequence[int], inplace=False) -> AnomalibDataset:
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
        return isinstance(self._samples, DataFrame)

    @property
    def samples(self) -> DataFrame:
        """Get the samples dataframe."""
        if not self.is_setup:
            raise RuntimeError("Dataset is not setup yet. Call setup() first.")
        return self._samples

    @samples.setter
    def samples(self, samples: DataFrame):
        """Overwrite the samples with a new dataframe.

        Args:
            samples (DataFrame): DataFrame with new samples.
        """
        # validate the passed samples by checking the
        assert isinstance(samples, DataFrame), f"samples must be a pandas.DataFrame, found {type(samples)}"
        expected_columns = _EXPECTED_COLS_PERTASK[self.task]
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

    def __getitem__(self, index: int) -> Dict[str, Union[str, Tensor]]:
        """Get dataset item for the index ``index``.

        Args:
            index (int): Index to get the item.

        Returns:
            Union[Dict[str, Tensor], Dict[str, Union[str, Tensor]]]: Dict of image tensor during training.
                Otherwise, Dict containing image path, target path, image tensor, label and transformed bounding box.
        """

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
