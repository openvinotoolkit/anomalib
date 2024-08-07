"""Dataset filtering with DatasetFilter.

This script contains the implementation of the DatasetFilter class,
which provides methods to filter a pandas DataFrame based on different criteria
such as label, indices, ratio, or count.
It also supports applying multiple filters sequentially.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any, cast

import pandas as pd

from anomalib.data.utils.label import LabelName


class DatasetFilter:
    """Filters the dataset based on various criteria.

    This class provides methods to filter a pandas DataFrame based on different criteria
    such as label, indices, ratio, or count. It also supports applying multiple filters sequentially.

    Args:
        samples (pd.DataFrame): The input DataFrame to be filtered.

    Methods:
        apply(by, seed=None, label_aware=False, inplace=False): Apply the specified filter(s).

    Returns:
        pd.DataFrame: Filtered DataFrame

    Examples:
        # Initialize the filter with a DataFrame
        dataset_filter = DatasetFilter(your_dataframe)

        # Filter by label
        normal_samples = dataset_filter.apply(by=LabelName.NORMAL)
        anomalous_samples = dataset_filter.apply(by="abnormal")

        # Filter by ratio (keep 30% of samples randomly)
        subset_samples = dataset_filter.apply(by=0.3, seed=42)

        # Filter by ratio with label-awareness (always contains both labels)
        balanced_samples = dataset_filter.apply(by=0.3, label_aware=True, seed=42)

        # Filter by count (keep 100 random samples)
        hundred_samples = dataset_filter.apply(by=100, seed=42)

        # Filter by count with label-awareness (always contains both labels)
        balanced_samples = dataset_filter.apply(by=100, label_aware=True, seed=42)

        # Filter by specific indices
        selected_samples = dataset_filter.apply(by=[0, 10, 20, 30])

        # Apply multiple filters (50% of normal samples)
        filtered_samples = dataset_filter.apply(by={"label": LabelName.NORMAL, "ratio": 0.5}, seed=42)

        # Modify the dataset in-place
        dataset_filter.apply(by=LabelName.NORMAL, inplace=True)

        # Use label-aware filtering (maintains label proportions)
        label_aware_samples = dataset_filter.apply(by=0.5, label_aware=True, seed=42)

    Notes:
        - When using ratio (float) or count (int), the samples are selected randomly.
        - The ``seed`` parameter ensures reproducibility for random selections.
        - Multiple filters are applied sequentially in the order they appear in the dictionary.
        - The ``label_aware`` option maintains the proportion of labels when filtering by ratio or count.
        - The ``inplace`` option modifies the original DataFrame instead of returning a new one.
        - Valid labels are ``normal`` or ``abnormal`` (case-sensitive) or
            ``LabelName.NORMAL`` or ``LabelName.ABNORMAL``.
    """

    def __init__(self, samples: pd.DataFrame) -> None:
        self._samples = samples

    @property
    def samples(self) -> pd.DataFrame:
        """pd.DataFrame: The input DataFrame to be filtered."""
        return self._samples

    @samples.setter
    def samples(self, value: pd.DataFrame) -> None:
        self._samples = value

    def __call__(
        self,
        by: str | LabelName | Sequence[int] | int | float | dict[str, Any],
        seed: int | None = None,
        label_aware: bool = False,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Apply the specified filter(s) to the DataFrame."""
        return self.apply(by, seed, label_aware, inplace)

    def apply(
        self,
        by: str | LabelName | Sequence[int] | int | float | dict[str, Any],
        seed: int | None = None,
        label_aware: bool = False,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Apply the specified filter(s) to the DataFrame."""
        if isinstance(by, str | LabelName):
            result = self.by_label(by)
        elif isinstance(by, Sequence):
            if all(isinstance(x, int) for x in by):
                indices = cast(Sequence[int], by)
                result = self.by_indices(indices)
            else:
                msg = "When 'by' is a Sequence, all elements must be integers."
                raise TypeError(msg)
        elif isinstance(by, int):
            result = self.by_count(by, seed=seed, label_aware=label_aware)
        elif isinstance(by, float):
            result = self.by_ratio(by, seed=seed, label_aware=label_aware)
        elif isinstance(by, dict):
            result = self.by_multiple(by, seed=seed, label_aware=label_aware, inplace=inplace)
        else:
            msg = f"Invalid filter: {by}. Expected str, LabelName, Sequence[int], int, float, or dict."
            raise TypeError(msg)

        if inplace:
            self.samples = result

        return result

    def by_label(self, label: str | LabelName, samples: pd.DataFrame | None = None) -> pd.DataFrame:
        """Filter the DataFrame by label."""
        if samples is None:
            samples = self.samples

        if isinstance(label, str) and label not in ["normal", "abnormal"]:
            msg = f"Invalid label: {label}. Must be 'normal' or 'abnormal'."
            raise ValueError(msg)

        if isinstance(label, LabelName) and label not in [LabelName.NORMAL, LabelName.ABNORMAL]:
            msg = f"Invalid LabelName: {label}. Must be LabelName.NORMAL or LabelName.ABNORMAL."
            raise ValueError(msg)

        label_index = LabelName.NORMAL if label in [LabelName.NORMAL, "normal"] else LabelName.ABNORMAL
        return samples[samples.label_index == label_index].reset_index(drop=True)

    def by_indices(self, indices: Sequence[int], samples: pd.DataFrame | None = None) -> pd.DataFrame:
        """Filter the DataFrame by specific indices."""
        if samples is None:
            samples = self.samples

        if len(set(indices)) != len(indices):
            msg = "Duplicate indices are not allowed."
            raise ValueError(msg)
        return samples.iloc[list(indices)].reset_index(drop=True)

    def by_count(
        self,
        count: int,
        samples: pd.DataFrame | None = None,
        seed: int | None = None,
        label_aware: bool = False,
    ) -> pd.DataFrame:
        """Filter the DataFrame by count."""
        if samples is None:
            samples = self.samples

        if not (0 < count <= len(samples)):
            msg = f"Count must be between 1 and {len(samples)}."
            raise ValueError(msg)

        if label_aware and "label_index" in samples.columns:
            grouped = samples.groupby("label_index")

            # Calculate the number of samples for each label
            label_counts = grouped.size()
            label_proportions = label_counts / label_counts.sum()
            samples_per_label = (label_proportions * count).round().astype(int)

            # Adjust to ensure we get exactly 'count' samples
            samples_per_label[samples_per_label.idxmax()] += count - samples_per_label.sum()

            # Sample from each group
            samples_from_each_group = [
                group.sample(n=samples_per_label[label], random_state=seed) for label, group in grouped
            ]

            return pd.concat(samples_from_each_group).sample(frac=1, random_state=seed).reset_index(drop=True)

        return samples.sample(n=count, random_state=seed).reset_index(drop=True)

    def by_ratio(
        self,
        ratio: float,
        samples: pd.DataFrame | None = None,
        seed: int | None = None,
        label_aware: bool = False,
    ) -> pd.DataFrame:
        """Filter the DataFrame by ratio."""
        if not (0 < ratio <= 1):
            msg = "Ratio must be between 0 and 1."
            raise ValueError(msg)

        if samples is None:
            samples = self.samples

        if label_aware and "label_index" in samples.columns:
            grouped = samples.groupby("label_index")
            sampled_dfs = [group.sample(frac=ratio, random_state=seed) for _, group in grouped]
            return pd.concat(sampled_dfs).reset_index(drop=True)

        return samples.sample(frac=ratio, random_state=seed).reset_index(drop=True)

    def by_multiple(
        self,
        filters: dict[str, Any],
        samples: pd.DataFrame | None = None,
        seed: int | None = None,
        label_aware: bool = False,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Apply multiple filters sequentially."""
        if samples is None:
            samples = self.samples

        result = samples.copy()
        for key, value in filters.items():
            # if results is empty, raise an error
            if result.empty:
                msg = "No samples left to filter."
                raise ValueError(msg)

            if key == "label":
                result = self.by_label(value, samples=result)
            elif key == "ratio":
                result = self.by_ratio(value, samples=result, seed=seed, label_aware=label_aware)
            elif key == "count":
                result = self.by_count(value, samples=result, seed=seed, label_aware=label_aware)
            else:
                msg = f"Unknown filter key: {key}. Must be 'label', 'ratio', or 'count'."
                raise ValueError(msg)

        if inplace:
            self.samples = result

        return result
