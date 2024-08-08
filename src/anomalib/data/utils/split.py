"""Dataset Split Utils.

This module contains function in regards to splitting normal images in training set,
and creating validation sets from test sets.

These function are useful
    - when the test set does not contain any normal images.
    - when the dataset doesn't have a validation set.
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import math
import warnings
from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import torch

from anomalib.data.utils.filter import DatasetFilter
from anomalib.data.utils.label import LabelName

if TYPE_CHECKING:
    from anomalib import data

logger = logging.getLogger(__name__)


class Split(str, Enum):
    """Split of a subset."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class TestSplitMode(str, Enum):
    """DEPRECATED: Splitting mode used to obtain subset."""

    NONE = "none"
    FROM_DIR = "from_dir"
    SYNTHETIC = "synthetic"


class ValSplitMode(str, Enum):
    """DEPRECATED: Splitting mode used to obtain validation subset."""

    NONE = "none"
    SAME_AS_TEST = "same_as_test"
    FROM_TRAIN = "from_train"
    FROM_TEST = "from_test"
    SYNTHETIC = "synthetic"


class SplitMode(str, Enum):
    """Unified splitting mode for both test and validation subsets.

    This enum represents the available modes for splitting datasets.

    Attributes:
        SYNTHETIC: Generate synthetic data for splitting.
        PREDEFINED: Use a pre-defined split from an existing source.
        AUTO: Automatically determine the best splitting strategy.

    Example:
        >>> mode = SplitMode.AUTO
        >>> print(mode)
        SplitMode.AUTO
        >>> print(mode.value)
        'auto'
    """

    SYNTHETIC = "synthetic"
    PREDEFINED = "predefined"
    AUTO = "auto"


def resolve_split_mode(split_mode: str | TestSplitMode | ValSplitMode | SplitMode | None = None) -> SplitMode:
    """DEPRECATED: Resolve various split mode inputs to the current ``SplitMode`` enum.

    This function handles both current ``SplitMode`` values and legacy split mode inputs,
    resolving them to the appropriate ``SplitMode`` enum value. It provides backward
    compatibility for legacy inputs while supporting current usage.

    Warning:
        Usage with legacy split modes is deprecated and will be removed in
        version 1.3. Please update your code to use ``SplitMode`` directly when
        possible. Deprecation warnings are raised for all legacy inputs.

    Args:
        split_mode: The split mode value as a string, ``TestSplitMode``,
            ``ValSplitMode``, or ``SplitMode``.

    Returns:
        The resolved ``SplitMode`` enum value.

    Raises:
        TypeError: If the input value is not a string, ``SplitMode``,
            or a valid ``TestSplitMode``/``ValSplitMode`` enum member.
        ValueError: If the input value is not recognized as a valid split mode.

    Examples:
        >>> resolve_split_mode(TestSplitMode.NONE)  # Legacy input (deprecated)
        DeprecationWarning: The split mode TestSplitMode.NONE is deprecated. Use 'SplitMode.AUTO' instead.
        SplitMode.AUTO

        >>> resolve_split_mode(TestSplitMode.SYNTHETIC)  # Legacy input (deprecated)
        DeprecationWarning: The split mode TestSplitMode.SYNTHETIC is deprecated. Use 'SplitMode.SYNTHETIC' instead.
        SplitMode.SYNTHETIC

        >>> resolve_split_mode(ValSplitMode.FROM_TRAIN)  # Legacy input (deprecated)
        DeprecationWarning: The split mode ValSplitMode.FROM_TRAIN is deprecated. Use 'SplitMode.AUTO' instead.
        SplitMode.AUTO

        >>> resolve_split_mode(SplitMode.PREDEFINED)  # Current input (preferred)
        SplitMode.PREDEFINED

    Note:
        For legacy inputs, this function will be removed in version 1.3.
        Update your code to use ``SplitMode`` enum directly when possible:

        # Old code (deprecated):
        mode = resolve_split_mode("from_dir")

        # New code:
        mode = SplitMode.PREDEFINED
    """
    if split_mode is None:
        return SplitMode.AUTO

    if isinstance(split_mode, SplitMode):
        return split_mode

    if not isinstance(split_mode, str | TestSplitMode | ValSplitMode):
        msg = "Input must be a string or a valid TestSplitMode/ValSplitMode enum member"
        raise TypeError(msg)

    original_input = split_mode
    value = split_mode.value if isinstance(split_mode, TestSplitMode | ValSplitMode) else split_mode
    value = str(value).lower()

    # Check for new SplitMode values first
    if value in [mode.value.lower() for mode in SplitMode]:
        return SplitMode(value)

    # Legacy mapping for old modes
    mapping = {
        "none": SplitMode.AUTO,
        "from_dir": SplitMode.PREDEFINED,
        "synthetic": SplitMode.SYNTHETIC,
        "same_as_test": SplitMode.AUTO,
        "from_train": SplitMode.AUTO,
        "from_test": SplitMode.AUTO,
    }

    if value in mapping:
        resolved_mode = mapping[value]
        if isinstance(original_input, TestSplitMode | ValSplitMode):
            deprecated_mode = f"{original_input.__class__.__name__}.{original_input.name}"
        else:
            deprecated_mode = f"'{original_input}'"

        warnings.warn(
            f"The split mode {deprecated_mode} is deprecated. Use 'SplitMode.{resolved_mode.name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return resolved_mode

    msg = f"Unrecognized split mode: {split_mode}"
    raise ValueError(msg)


def concatenate_datasets(datasets: Sequence["data.AnomalibDataset"]) -> "data.AnomalibDataset":
    """Concatenate multiple datasets into a single dataset object.

    Args:
        datasets (Sequence[AnomalibDataset]): Sequence of at least two datasets.

    Returns:
        AnomalibDataset: Dataset that contains the combined samples of all input datasets.
    """
    concat_dataset = datasets[0]
    for dataset in datasets[1:]:
        concat_dataset += dataset
    return concat_dataset


def random_split(
    dataset: "data.AnomalibDataset",
    split_ratio: float | Sequence[float],
    label_aware: bool = False,
    seed: int | None = None,
) -> list["data.AnomalibDataset"]:
    """Perform a random split of a dataset.

    Args:
        dataset (AnomalibDataset): Source dataset
        split_ratio (Union[float, Sequence[float]]): Fractions of the splits that will be produced. The values in the
            sequence must sum to 1. If a single value is passed, the ratio will be converted to
            [1-split_ratio, split_ratio].
        label_aware (bool): When True, the relative occurrence of the different class labels of the source dataset will
            be maintained in each of the subsets.
        seed (int | None, optional): Seed that can be passed if results need to be reproducible
    """
    if isinstance(split_ratio, float):
        split_ratio = [1 - split_ratio, split_ratio]

    if not (math.isclose(sum(split_ratio), 1) and sum(split_ratio) <= 1):
        msg = f"Split ratios must sum to 1, found {sum(split_ratio)}"
        raise ValueError(msg)

    if not all(0 < ratio < 1 for ratio in split_ratio):
        msg = f"All split ratios must be between 0 and 1, found {split_ratio}"
        raise ValueError(msg)

    # create list of source data
    if label_aware and "label_index" in dataset.samples:
        indices_per_label = [group.index for _, group in dataset.samples.groupby("label_index")]
        per_label_datasets = [dataset.subsample(indices) for indices in indices_per_label]
    else:
        per_label_datasets = [dataset]

    # outer list: per-label unique, inner list: random subsets with the given ratio
    subsets: list[list[data.AnomalibDataset]] = []
    # split each (label-aware) subset of source data
    for label_dataset in per_label_datasets:
        # get subset lengths
        subset_lengths = [math.floor(len(label_dataset.samples) * ratio) for ratio in split_ratio]
        for i in range(len(label_dataset.samples) - sum(subset_lengths)):
            subset_idx = i % sum(subset_lengths)
            subset_lengths[subset_idx] += 1
        if 0 in subset_lengths:
            msg = """Zero subset length encountered during splitting. This means one of your subsets
            might be empty or devoid of either normal or anomalous images."""
            logger.warning(msg)

        # perform random subsampling
        random_state = torch.Generator().manual_seed(seed) if seed else None
        indices = torch.randperm(len(label_dataset.samples), generator=random_state)
        subsets.append(
            [label_dataset.subsample(subset_indices) for subset_indices in torch.split(indices, subset_lengths)],
        )

    # invert outer/inner lists
    # outer list: subsets with the given ratio, inner list: per-label unique
    subsets = list(map(list, zip(*subsets, strict=True)))
    return [concatenate_datasets(subset) for subset in subsets]


def split_by_indices(
    dataset: "data.AnomalibDataset",
    indices: Sequence[int],
    inplace: bool = False,
) -> "data.AnomalibDataset":
    """Split the dataset by selecting specific indices.

    Args:
        dataset (AnomalibDataset): Source dataset
        indices (Sequence[int]): Specific indices to select
        inplace (bool): When true, the split will be performed on the instance itself.
            Defaults to False.

    Returns:
        AnomalibDataset: Dataset containing only the selected indices.
    """
    if len(set(indices)) != len(indices):
        msg = "No duplicates allowed in indices."
        raise ValueError(msg)

    split_dataset = dataset if inplace else copy.deepcopy(dataset)
    split_dataset.samples = dataset.samples.iloc[indices].reset_index(drop=True)
    return split_dataset


def split_by_label(dataset: "data.AnomalibDataset") -> tuple["data.AnomalibDataset", "data.AnomalibDataset"]:
    """Split the dataset into the normal and anomalous subsets."""
    samples = dataset.samples
    normal_indices = samples[samples.label_index == 0].index
    anomalous_indices = samples[samples.label_index == 1].index

    normal_subset = dataset.subsample(list(normal_indices))
    anomalous_subset = dataset.subsample(list(anomalous_indices))
    return normal_subset, anomalous_subset
    return normal_subset, anomalous_subset


class SubsetCreator:
    """Create subsets from a DataFrame based on various criteria."""

    def __init__(self, samples: pd.DataFrame) -> None:
        self.samples = samples
        self.filter = DatasetFilter(samples)

    def create(
        self,
        criteria: Literal["label"] | Sequence[int] | Sequence[float] | int | float | dict[str, Any],
        seed: int | None = None,
        label_aware: bool = False,
    ) -> list[pd.DataFrame]:
        """Apply the specified split operation to the DataFrame.

        Args:
            criteria: Splitting criteria. Can be "label", a sequence of indices, a sequence of floats (ratios),
                an integer (count), or a dictionary for mixed criteria.
            seed: Random seed for reproducibility when using random sampling.
            label_aware: If True, maintain label proportions in splits where applicable.

        Returns:
            List[pd.DataFrame]: List of split DataFrames

        Examples:
            # Split by label
            normal, abnormal = subset_creator.create("label")

            # Split by indices
            train, val, test = subset_creator.create([[0, 1, 2], [3, 4], [5, 6, 7]])

            # Split by ratio
            train, val, test = subset_creator.create([0.7, 0.2, 0.1], label_aware=True, seed=42)

            # Split by count
            train, val, test = subset_creator.create([700, 200, 100], label_aware=True, seed=42)

            # Split by mixed criteria
            normal_train, abnormal_subset, remaining = subset_creator.create(
                [{"label": LabelName.NORMAL, "ratio": 0.7},
                 {"label": LabelName.ABNORMAL, "count": 100}],
                seed=42,
                label_aware=True
            )
        """
        if criteria == "label":
            splits = self.split_by_label()
        if isinstance(criteria, int):
            splits = self.split_by_count(criteria, label_aware=label_aware, seed=seed)
        if isinstance(criteria, float):
            splits = self.split_by_ratio(criteria, label_aware=label_aware, seed=seed)
        if isinstance(criteria, dict):
            splits = self.split_by_mixed_criteria(criteria, label_aware=label_aware, seed=seed)
        if isinstance(criteria, Sequence):
            if all(isinstance(x, int) for x in criteria):
                # eg., [100, 200, 300]
                splits = self.split_by_count(*criteria, label_aware=label_aware, seed=seed)
            elif all(isinstance(x, float) for x in criteria):
                # eg., [0.7, 0.2, 0.1]
                splits = self.split_by_ratio(*criteria, label_aware=label_aware, seed=seed)
            elif all(isinstance(x, Sequence) and all(isinstance(y, int) for y in x) for x in criteria):
                # eg., [[0, 1, 2], [3, 4], [5, 6, 7]]
                splits = self.split_by_indices(*criteria)
            elif all(isinstance(x, dict) for x in criteria):
                # eg., [{"label": "normal", "ratio": 0.7}, {"label": "abnormal", "count": 100}]
                splits = self.split_by_mixed_criteria(*criteria, seed=seed, label_aware=label_aware)
            else:
                msg = f"Invalid sequence type for splitting: {criteria}"
                raise ValueError(msg)

        if splits is None:
            msg = f"Invalid criteria for splitting: {criteria}. "
            raise ValueError(msg)

        return splits

    def __call__(
        self,
        by: Literal["label"] | Sequence[int] | Sequence[float] | int | dict[str, Any],
        seed: int | None = None,
        label_aware: bool = False,
    ) -> list[pd.DataFrame]:
        """Calls the apply method to perform the split operation."""
        return self.create(by, seed=seed, label_aware=label_aware)

    def split_by_label(self) -> list[pd.DataFrame]:
        """Split the dataset by label."""
        if "label_index" not in self.samples.columns:
            msg = "Cannot split by label: 'label_index' column not found in the DataFrame"
            raise ValueError(msg)
        return [self.filter.by_label(label, self.samples) for label in [LabelName.NORMAL, LabelName.ABNORMAL]]

    def split_by_indices(self, *indices: Sequence[int]) -> list[pd.DataFrame]:
        """Split the dataset by multiple lists of indices."""
        all_indices = [idx for sublist in indices for idx in sublist]
        if len(set(all_indices)) != len(all_indices):
            msg = "Overlapping indices detected in the provided index lists."
            raise ValueError(msg)

        return [self.filter.by_indices(indices, self.samples) for indices in indices]

    def split_by_count(self, *counts: int, label_aware: bool = False, seed: int | None = None) -> list[pd.DataFrame]:
        """Split the dataset into multiple parts based on specified counts."""
        total_count = sum(counts)
        if total_count > len(self.samples):
            msg = f"Sum of counts ({total_count}) exceeds total number of samples ({len(self.samples)})."
            raise ValueError(msg)

        splits = []
        remaining_samples = self.samples.copy()

        for count in counts:
            split = self.filter.by_count(count, remaining_samples, seed, label_aware)
            splits.append(split)
            remaining_samples = remaining_samples[~remaining_samples.index.isin(split.index)]

        return splits

    def split_by_ratio(
        self,
        *ratios: float,
        label_aware: bool = False,
        seed: None | int = None,
    ) -> list[pd.DataFrame]:
        """Split the dataset into multiple parts based on specified ratios."""
        if not all(0 < ratio < 1 for ratio in ratios):
            msg = "All ratios must be between 0 and 1."
            raise ValueError(msg)

        if sum(ratios) > 1:
            msg = "Sum of ratios cannot exceed 1."
            raise ValueError(msg)

        total_samples = len(self.samples)
        splits = []
        remaining_samples = self.samples.copy()

        for ratio in ratios:
            split_size = int(total_samples * ratio)
            split = self.filter.by_count(split_size, remaining_samples, seed, label_aware)
            splits.append(split)
            remaining_samples = remaining_samples[~remaining_samples.index.isin(split.index)]

        return splits

    def split_by_mixed_criteria(
        self,
        *criteria: dict[str, Any],
        seed: int | None = None,
        label_aware: bool = False,
    ) -> list[pd.DataFrame]:
        """Split the dataset into multiple parts based on multiple filtering criteria.

        Args:
            *criteria: Variable number of dictionaries, each containing filtering criteria for one split.
            seed: Random seed for reproducibility.
            label_aware: If True, maintain label proportions in splits where applicable.

        Returns:
            List[pd.DataFrame]: List of DataFrames, each filtered according to the provided criteria.

        Example:
            split1, split2 = subset_creator.split_by_mixed_criteria(
                {"label": LabelName.NORMAL, "ratio": 0.7},
                {"label": LabelName.ABNORMAL, "count": 100},
                seed=42,
                label_aware=True
            )
        """
        splits = []
        remaining_samples = self.samples.copy()

        for criterion in criteria:
            split = self.filter.by_multiple(criterion, remaining_samples, seed, label_aware)
            splits.append(split)

            merged_df = remaining_samples.merge(split, how="left", indicator=True)
            remaining_samples = merged_df[merged_df["_merge"] == "left_only"].drop("_merge", axis=1)

        return splits
