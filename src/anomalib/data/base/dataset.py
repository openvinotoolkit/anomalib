"""Anomalib dataset base class."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import warnings
from abc import ABC
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform
from torchvision.tv_tensors import Mask

from anomalib import TaskType
from anomalib.data.utils import LabelName, masks_to_boxes, read_image, read_mask
from anomalib.data.utils.filter import DatasetFilter
from anomalib.data.utils.split import SubsetCreator

_EXPECTED_COLUMNS_CLASSIFICATION = ["image_path"]
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
        self._filter: DatasetFilter | None = None
        self._subset_creator: SubsetCreator | None = None

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
        warnings.warn(
            "The 'subsample' method is deprecated and will be removed in a future version. "
            "Use 'subset' method instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        filtered_samples = self.filter(indices)
        if inplace:
            self.samples = filtered_samples
            return self

        return self._create_subset(filtered_samples)

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
    def filter(self) -> DatasetFilter:
        """Get the dataset filter.

        Returns:
            DatasetFilter: Dataset filter instance.

        Examples:
            Apply filters to the dataset:
            >>> dataset.filter.by_label("normal")
            >>> dataset.filter.by_count(100)
            >>> dataset.filter.by_ratio(0.5)

            Apply multiple filters:
            >>> dataset.filter.by_multiple({"label": "normal", "count": 100})

            Apply filters in place:
            >>> dataset.filter.by_label("normal", inplace=True)

            Apply filters via apply method:
            >>> dataset.filter.apply("normal")  # label
            >>> dataset.filter.apply(100)       # count
            >>> dataset.filter.apply(0.5)       # ratio
            >>> dataset.filter.apply({"label": "normal", "count": 100})  # multiple filters

            Similarly __call__ method can be used to apply filters:
            >>> dataset.filter("normal")    # label
            >>> dataset.filter(100)         # count
            >>> dataset.filter(0.5)         # ratio
            >>> dataset.filter({"label": "normal", "count": 100}) # multiple filters
        """
        if self._filter is None:
            self._filter = DatasetFilter(self.samples)
        return self._filter

    def create_subset(
        self,
        criteria: Literal["label"] | Sequence[int] | Sequence[float] | int | float | dict[str, Any],
        seed: int | None = None,
        label_aware: bool = False,
    ) -> list["AnomalibDataset"]:
        """Create subsets of the current dataset based on given criteria.

        Args:
            criteria (Literal["label"] | Sequence[int] | Sequence[float] | int | dict[str, Any]):
                The criteria used to create subsets. It can be one of the following types:
                - ``label``: Subset based on label values.
                - ``Sequence[int]``: Subset based on specific sample indices.
                - ``Sequence[float]``: Subset based on specific sample weights.
                - ``int``: Subset based on the number of samples.
                - ``dict[str, Any]``: Subset based on custom criteria.

            seed (int | None, optional):
                The random seed used for creating subsets.
                    Defaults to ``None``.

            label_aware (bool, optional):
                Whether to create subsets while preserving the label distribution.
                    Defaults to ``False``.

        Returns:
            list["AnomalibDataset"]: A list of AnomalibDataset objects representing the subsets.

        Examples:
            Create a subset based on label values:
            >>> normal_dataset, abnormal_dataset = dataset.create_subset("label")

            Create a subset based on specific sample indices:
            >>> train_set, val_set, test_set = dataset.create_subset([[0, 2, 3], [1, 4], [5]])

            Create a subset based on specific split ratios:
            >>> train_set, val_set, test_set = dataset.create_subset([0.6, 0.2, 0.2], seed=42)

            Create a subset based on the number of samples:
            >>> dataset.create_subset(100)

            Create a subset based on custom criteria:
            >>> dataset.create_subset({"label": "normal", "count": 100})
        """
        subset_samples = self.subset.create(criteria, seed=seed, label_aware=label_aware)
        return [self._create_subset(samples) for samples in subset_samples]

    @property
    def subset(self) -> SubsetCreator:
        """Get the subset creator."""
        if self._subset_creator is None:
            self._subset_creator = SubsetCreator(self.samples)
        return self._subset_creator

    def _create_subset(self, samples: pd.DataFrame) -> "AnomalibDataset":
        subset = copy.deepcopy(self)
        subset.samples = samples
        return subset

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

    @property
    def all_normal(self) -> bool:
        """Check if all samples in the dataset are normal."""
        return self.has_normal and not self.has_anomalous

    @property
    def all_anomalous(self) -> bool:
        """Check if all samples in the dataset are anomalous."""
        return self.has_anomalous and not self.has_normal

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
        elif self.task in (TaskType.DETECTION, TaskType.SEGMENTATION):
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

    def copy(self) -> "AnomalibDataset":
        """Create a deep copy of the dataset.

        Returns:
            AnomalibDataset: A new instance of the dataset with the same attributes.
        """
        return copy.deepcopy(self)

    # Alias for copy method
    clone = copy
