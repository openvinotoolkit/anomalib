"""Anomalib dataset base class.

This module provides the base dataset class for Anomalib datasets. The dataset is based on a
dataframe that contains the information needed by the dataloader to load each dataset item
into memory.

The samples dataframe must be set from the subclass using the setter of the ``samples``
property.

The DataFrame must include at least the following columns:
    - ``split`` (str): The subset to which the dataset item is assigned (e.g., 'train',
      'test').
    - ``image_path`` (str): Path to the file system location where the image is stored.
    - ``label_index`` (int): Index of the anomaly label, typically 0 for 'normal' and 1 for
      'anomalous'.
    - ``mask_path`` (str, optional): Path to the ground truth masks (for anomalous images
      only). Required if task is 'segmentation'.

Example DataFrame:
    >>> df = pd.DataFrame({
    ...     'image_path': ['path/to/image.png'],
    ...     'label': ['anomalous'],
    ...     'label_index': [1],
    ...     'mask_path': ['path/to/mask.png'],
    ...     'split': ['train']
    ... })
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
from abc import ABC
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform
from torchvision.tv_tensors import Mask

from anomalib import TaskType
from anomalib.data.dataclasses import DatasetItem, ImageBatch, ImageItem
from anomalib.data.utils import LabelName, read_image, read_mask

_EXPECTED_COLUMNS = ["image_path", "split"]

logger = logging.getLogger(__name__)


class AnomalibDataset(Dataset, ABC):
    """Base class for Anomalib datasets.

    The dataset is designed to work with image-based anomaly detection tasks. It supports
    both classification and segmentation tasks.

    Args:
        transform (Transform | None, optional): Transforms to be applied to the input images.
            Defaults to ``None``.

    Example:
        >>> from torchvision.transforms.v2 import Resize
        >>> dataset = AnomalibDataset(transform=Resize((256, 256)))
        >>> len(dataset)  # Get dataset length
        100
        >>> item = dataset[0]  # Get first item
        >>> item.image.shape
        torch.Size([3, 256, 256])

    Note:
        The example above is illustrative and may need to be adjusted based on the specific dataset structure.

    Args:
        task (str): Task type, either 'classification' or 'segmentation'
        augmentations (Transform, optional): Augmentations that should be applied to the input images.
            Defaults to ``None``.
    """

    def __init__(self, augmentations: Transform | None = None) -> None:
        super().__init__()
        self.augmentations = augmentations
        self._samples: DataFrame | None = None
        self._category: str | None = None

    @property
    def name(self) -> str:
        """Get the name of the dataset.

        Returns:
            str: Name of the dataset derived from the class name, with 'Dataset' suffix
                removed if present.

        Example:
            >>> dataset = AnomalibDataset()
            >>> dataset.name
            'Anomalib'
        """
        class_name = self.__class__.__name__

        # Remove the `_dataset` suffix from the class name
        if class_name.endswith("Dataset"):
            class_name = class_name[:-7]

        return class_name

    def __len__(self) -> int:
        """Get length of the dataset.

        Returns:
            int: Number of samples in the dataset.

        Raises:
            RuntimeError: If samples DataFrame is not set.
        """
        return len(self.samples)

    def subsample(self, indices: Sequence[int], inplace: bool = False) -> "AnomalibDataset":
        """Create a subset of the dataset using the provided indices.

        Args:
            indices (Sequence[int]): Indices at which the dataset is to be subsampled.
            inplace (bool, optional): When true, modify the instance itself. Defaults to
                ``False``.

        Returns:
            AnomalibDataset: Subsampled dataset.

        Raises:
            ValueError: If duplicate indices are provided.

        Example:
            >>> dataset = AnomalibDataset()
            >>> subset = dataset.subsample([0, 1, 2])
            >>> len(subset)
            3
        """
        if len(set(indices)) != len(indices):
            msg = "No duplicates allowed in indices."
            raise ValueError(msg)
        dataset = self if inplace else copy.deepcopy(self)
        dataset.samples = self.samples.iloc[indices].reset_index(drop=True)
        return dataset

    @property
    def samples(self) -> DataFrame:
        """Get the samples DataFrame.

        Returns:
            DataFrame: DataFrame containing dataset samples.

        Raises:
            RuntimeError: If samples DataFrame has not been set.
        """
        if self._samples is None:
            msg = (
                "Dataset does not have a samples dataframe. Ensure that a dataframe has "
                "been assigned to `dataset.samples`."
            )
            raise RuntimeError(msg)
        return self._samples

    @samples.setter
    def samples(self, samples: DataFrame) -> None:
        """Set the samples DataFrame.

        Args:
            samples (DataFrame): DataFrame containing dataset samples.

        Raises:
            TypeError: If samples is not a pandas DataFrame.
            ValueError: If required columns are missing.
            FileNotFoundError: If any image paths do not exist.

        Example:
            >>> df = pd.DataFrame({
            ...     'image_path': ['image.png'],
            ...     'split': ['train']
            ... })
            >>> dataset = AnomalibDataset()
            >>> dataset.samples = df
        """
        # validate the passed samples by checking the
        if not isinstance(samples, DataFrame):
            msg = f"samples must be a pandas.DataFrame, found {type(samples)}"
            raise TypeError(msg)

        if not all(col in samples.columns for col in _EXPECTED_COLUMNS):
            msg = f"samples must have (at least) columns {_EXPECTED_COLUMNS}, found {samples.columns}"
            raise ValueError(msg)

        if not samples["image_path"].apply(lambda p: Path(p).exists()).all():
            msg = "missing file path(s) in samples"
            raise FileNotFoundError(msg)

        self._samples = samples.sort_values(by="image_path", ignore_index=True)

    @property
    def category(self) -> str | None:
        """Get the category of the dataset.

        Returns:
            str | None: Dataset category if set, else None.
        """
        return self._category

    @category.setter
    def category(self, category: str) -> None:
        """Set the category of the dataset.

        Args:
            category (str): Category to assign to the dataset.
        """
        self._category = category

    @property
    def has_normal(self) -> bool:
        """Check if the dataset contains normal samples.

        Returns:
            bool: True if dataset contains normal samples, False otherwise.
        """
        return LabelName.NORMAL in list(self.samples.label_index)

    @property
    def has_anomalous(self) -> bool:
        """Check if the dataset contains anomalous samples.

        Returns:
            bool: True if dataset contains anomalous samples, False otherwise.
        """
        return LabelName.ABNORMAL in list(self.samples.label_index)

    @property
    def task(self) -> TaskType:
        """Get the task type from the dataset.

        Returns:
            TaskType: Type of task (classification or segmentation).

        Raises:
            ValueError: If task type is unknown.
        """
        return TaskType(self.samples.attrs["task"])

    def __getitem__(self, index: int) -> DatasetItem:
        """Get dataset item for the given index.

        Args:
            index (int): Index to get the item.

        Returns:
            DatasetItem: Dataset item containing image and ground truth (if available).

        Example:
            >>> dataset = AnomalibDataset()
            >>> item = dataset[0]
            >>> isinstance(item.image, torch.Tensor)
            True
        """
        image_path = self.samples.iloc[index].image_path
        mask_path = self.samples.iloc[index].mask_path
        label_index = self.samples.iloc[index].label_index

        # Read the image
        image = read_image(image_path, as_tensor=True)

        # Initialize mask as None
        gt_mask = None

        # Process based on task type
        if self.task == TaskType.SEGMENTATION:
            if label_index == LabelName.NORMAL:
                # Create zero mask for normal samples
                gt_mask = Mask(torch.zeros(image.shape[-2:])).to(torch.uint8)
            elif label_index == LabelName.ABNORMAL:
                # Read mask for anomalous samples
                gt_mask = read_mask(mask_path, as_tensor=True)
            # For UNKNOWN, gt_mask remains None

        # Apply augmentations if available
        if self.augmentations:
            if self.task == TaskType.CLASSIFICATION:
                image = self.augmentations(image)
            elif self.task == TaskType.SEGMENTATION:
                # For augmentations that require both image and mask:
                # - Use a temporary zero mask for UNKNOWN samples
                # - But preserve the final gt_mask as None for UNKNOWN
                temp_mask = gt_mask if gt_mask is not None else Mask(torch.zeros(image.shape[-2:])).to(torch.uint8)
                image, augmented_mask = self.augmentations(image, temp_mask)
                # Only update gt_mask if it wasn't None before augmentations
                if gt_mask is not None:
                    gt_mask = augmented_mask

        # Create gt_label tensor (None for UNKNOWN)
        gt_label = None if label_index == LabelName.UNKNOWN else torch.tensor(label_index)

        # Return the dataset item
        return ImageItem(
            image=image,
            gt_mask=gt_mask,
            gt_label=gt_label,
            image_path=image_path,
            mask_path=mask_path,
        )

    def __add__(self, other_dataset: "AnomalibDataset") -> "AnomalibDataset":
        """Concatenate this dataset with another dataset.

        Args:
            other_dataset (AnomalibDataset): Dataset to concatenate with.

        Returns:
            AnomalibDataset: Concatenated dataset.

        Raises:
            TypeError: If datasets are not of the same type.

        Example:
            >>> dataset1 = AnomalibDataset()
            >>> dataset2 = AnomalibDataset()
            >>> combined = dataset1 + dataset2
        """
        if not isinstance(other_dataset, self.__class__):
            msg = "Cannot concatenate datasets that are not of the same type."
            raise TypeError(msg)
        dataset = copy.deepcopy(self)
        dataset.samples = pd.concat([self.samples, other_dataset.samples], ignore_index=True)
        return dataset

    @property
    def collate_fn(self) -> Callable:
        """Get the collate function for batching dataset items.

        Returns:
            Callable: Collate function from ImageBatch.

        Note:
            By default, this returns ImageBatch's collate function. Override this property
            for other dataset types.
        """
        return ImageBatch.collate
