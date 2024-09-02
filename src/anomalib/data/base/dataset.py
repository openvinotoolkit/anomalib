"""Anomalib dataset base class."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
from abc import ABC
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform
from torchvision.tv_tensors import Mask

from anomalib import TaskType
from anomalib.data.utils import LabelName, masks_to_boxes, read_image, read_mask

_EXPECTED_COLUMNS_CLASSIFICATION = ["image_path", "split"]
_EXPECTED_COLUMNS_SEGMENTATION = [*_EXPECTED_COLUMNS_CLASSIFICATION, "mask_path"]
_EXPECTED_COLUMNS_PERTASK = {
    "classification": _EXPECTED_COLUMNS_CLASSIFICATION,
    "segmentation": _EXPECTED_COLUMNS_SEGMENTATION,
    "detection": _EXPECTED_COLUMNS_SEGMENTATION,
}

logger = logging.getLogger(__name__)


class AnomalibDataset(Dataset, ABC):
    """Anomalib dataset.

    The dataset is based on a dataframe that contains the information needed by the dataloader to load each of
    the dataset items into memory.

    The samples dataframe must be set from the subclass using the setter of the `samples` property.

    The DataFrame must, at least, include the following columns:
        - `split` (str): The subset to which the dataset item is assigned (e.g., 'train', 'test').
        - `image_path` (str): Path to the file system location where the image is stored.
        - `label_index` (int): Index of the anomaly label, typically 0 for 'normal' and 1 for 'anomalous'.
        - `mask_path` (str, optional): Path to the ground truth masks (for the anomalous images only).
        Required if task is 'segmentation'.

    Example DataFrame:
        +---+-------------------+-----------+-------------+------------------+-------+
        |   | image_path        | label     | label_index | mask_path        | split |
        +---+-------------------+-----------+-------------+------------------+-------+
        | 0 | path/to/image.png | anomalous | 1           | path/to/mask.png | train |
        +---+-------------------+-----------+-------------+------------------+-------+

    Note:
        The example above is illustrative and may need to be adjusted based on the specific dataset structure.

    Args:
        task (str): Task type, either 'classification' or 'segmentation'
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
    """

    def __init__(self, task: TaskType | str, transform: Transform | None = None) -> None:
        super().__init__()
        self.task = TaskType(task)
        self.transform = transform
        self._samples: DataFrame | None = None
        self._category: str | None = None

    @property
    def name(self) -> str:
        """Name of the dataset."""
        class_name = self.__class__.__name__

        # Remove the `_dataset` suffix from the class name
        if class_name.endswith("Dataset"):
            class_name = class_name[:-7]

        return class_name

    def __len__(self) -> int:
        """Get length of the dataset."""
        return len(self.samples)

    def subsample(self, indices: Sequence[int], inplace: bool = False) -> "AnomalibDataset":
        """Subsamples the dataset at the provided indices.

        Args:
            indices (Sequence[int]): Indices at which the dataset is to be subsampled.
            inplace (bool): When true, the subsampling will be performed on the instance itself.
                Defaults to ``False``.
        """
        if len(set(indices)) != len(indices):
            msg = "No duplicates allowed in indices."
            raise ValueError(msg)
        dataset = self if inplace else copy.deepcopy(self)
        dataset.samples = self.samples.iloc[indices].reset_index(drop=True)
        return dataset

    @property
    def samples(self) -> DataFrame:
        """Get the samples dataframe."""
        if self._samples is None:
            msg = (
                "Dataset does not have a samples dataframe. Ensure that a dataframe has been assigned to "
                "`dataset.samples`."
            )
            raise RuntimeError(msg)
        return self._samples

    @samples.setter
    def samples(self, samples: DataFrame) -> None:
        """Overwrite the samples with a new dataframe.

        Args:
            samples (DataFrame): DataFrame with new samples.
        """
        # validate the passed samples by checking the
        if not isinstance(samples, DataFrame):
            msg = f"samples must be a pandas.DataFrame, found {type(samples)}"
            raise TypeError(msg)

        expected_columns = _EXPECTED_COLUMNS_PERTASK[self.task]
        if not all(col in samples.columns for col in expected_columns):
            msg = f"samples must have (at least) columns {expected_columns}, found {samples.columns}"
            raise ValueError(msg)

        if not samples["image_path"].apply(lambda p: Path(p).exists()).all():
            msg = "missing file path(s) in samples"
            raise FileNotFoundError(msg)

        self._samples = samples.sort_values(by="image_path", ignore_index=True)

    @property
    def category(self) -> str | None:
        """Get the category of the dataset."""
        return self._category

    @category.setter
    def category(self, category: str) -> None:
        """Set the category of the dataset."""
        self._category = category

    @property
    def has_normal(self) -> bool:
        """Check if the dataset contains any normal samples."""
        return LabelName.NORMAL in list(self.samples.label_index)

    @property
    def has_anomalous(self) -> bool:
        """Check if the dataset contains any anomalous samples."""
        return LabelName.ABNORMAL in list(self.samples.label_index)

    def __getitem__(self, index: int) -> dict[str, str | torch.Tensor]:
        """Get dataset item for the index ``index``.

        Args:
            index (int): Index to get the item.

        Returns:
            dict[str, str | torch.Tensor]: Dict of image tensor during training. Otherwise, Dict containing image path,
                target path, image tensor, label and transformed bounding box.
        """
        image_path = self.samples.iloc[index].image_path
        mask_path = self.samples.iloc[index].mask_path
        label_index = self.samples.iloc[index].label_index

        image = read_image(image_path, as_tensor=True)
        item = {"image_path": image_path, "label": label_index}

        if self.task == TaskType.CLASSIFICATION:
            item["image"] = self.transform(image) if self.transform else image
        elif self.task in {TaskType.DETECTION, TaskType.SEGMENTATION}:
            # Only Anomalous (1) images have masks in anomaly datasets
            # Therefore, create empty mask for Normal (0) images.
            mask = (
                Mask(torch.zeros(image.shape[-2:])).to(torch.uint8)
                if label_index == LabelName.NORMAL
                else read_mask(mask_path, as_tensor=True)
            )
            item["image"], item["mask"] = self.transform(image, mask) if self.transform else (image, mask)

            if self.task == TaskType.DETECTION:
                # create boxes from masks for detection task
                boxes, _ = masks_to_boxes(item["mask"])
                item["boxes"] = boxes[0]
        else:
            msg = f"Unknown task type: {self.task}"
            raise ValueError(msg)

        return item

    def __add__(self, other_dataset: "AnomalibDataset") -> "AnomalibDataset":
        """Concatenate this dataset with another dataset.

        Args:
            other_dataset (AnomalibDataset): Dataset to concatenate with.

        Returns:
            AnomalibDataset: Concatenated dataset.
        """
        if not isinstance(other_dataset, self.__class__):
            msg = "Cannot concatenate datasets that are not of the same type."
            raise TypeError(msg)
        dataset = copy.deepcopy(self)
        dataset.samples = pd.concat([self.samples, other_dataset.samples], ignore_index=True)
        return dataset
