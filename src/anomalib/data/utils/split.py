"""Dataset splitting utilities.

This module provides functions for splitting datasets in anomaly detection tasks:

- Splitting normal images into training and validation sets
- Creating validation sets from test sets
- Label-aware splitting to maintain class distributions
- Random splitting with optional seed for reproducibility

These utilities are particularly useful when:

- The test set lacks normal images
- The dataset needs a validation set
- Class balance needs to be maintained during splits

Example:
    >>> from anomalib.data.utils.split import random_split
    >>> # Split dataset with 80/20 ratio
    >>> train_set, val_set = random_split(dataset, split_ratio=0.2)
    >>> len(train_set), len(val_set)
    (800, 200)

    >>> # Label-aware split preserving class distributions
    >>> splits = random_split(dataset, [0.7, 0.2, 0.1], label_aware=True)
    >>> len(splits)
    3
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from anomalib.data import datasets as data

logger = logging.getLogger(__name__)


class Split(str, Enum):
    """Dataset split type.

    Attributes:
        TRAIN: Training split
        VAL: Validation split
        TEST: Test split
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class TestSplitMode(str, Enum):
    """Mode used to obtain test split.

    Attributes:
        NONE: No test split
        FROM_DIR: Test split from directory
        SYNTHETIC: Synthetic test split
    """

    NONE = "none"
    FROM_DIR = "from_dir"
    SYNTHETIC = "synthetic"


class ValSplitMode(str, Enum):
    """Mode used to obtain validation split.

    Attributes:
        NONE: No validation split
        SAME_AS_TEST: Use same split as test
        FROM_TRAIN: Split from training set
        FROM_TEST: Split from test set
        SYNTHETIC: Synthetic validation split
        FROM_DIR: Use dedicated validation directory (for datasets that have one)
    """

    NONE = "none"
    SAME_AS_TEST = "same_as_test"
    FROM_TRAIN = "from_train"
    FROM_TEST = "from_test"
    SYNTHETIC = "synthetic"
    FROM_DIR = "from_dir"


def concatenate_datasets(
    datasets: Sequence["data.AnomalibDataset"],
) -> "data.AnomalibDataset":
    """Concatenate multiple datasets into a single dataset.

    Args:
        datasets: Sequence of at least two datasets to concatenate

    Returns:
        Combined dataset containing samples from all input datasets

    Example:
        >>> combined = concatenate_datasets([dataset1, dataset2])
        >>> len(combined) == len(dataset1) + len(dataset2)
        True
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
    """Randomly split a dataset into multiple subsets.

    Args:
        dataset: Source dataset to split
        split_ratio: Split ratios that must sum to 1. If single float ``x`` is
            provided, splits into ``[1-x, x]``
        label_aware: If ``True``, maintains class label distributions in splits
        seed: Random seed for reproducibility

    Returns:
        List of dataset splits based on provided ratios

    Example:
        >>> splits = random_split(dataset, [0.7, 0.3], seed=42)
        >>> len(splits)
        2
        >>> # Label-aware splitting
        >>> splits = random_split(dataset, 0.2, label_aware=True)
        >>> len(splits)
        2
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


def split_by_label(
    dataset: "data.AnomalibDataset",
) -> tuple["data.AnomalibDataset", "data.AnomalibDataset"]:
    """Split dataset into normal and anomalous subsets.

    Args:
        dataset: Dataset to split by label

    Returns:
        Tuple containing:
            - Dataset with only normal samples (label 0)
            - Dataset with only anomalous samples (label 1)

    Example:
        >>> normal, anomalous = split_by_label(dataset)
        >>> len(normal) + len(anomalous) == len(dataset)
        True
    """
    samples = dataset.samples
    normal_indices = samples[samples.label_index == 0].index
    anomalous_indices = samples[samples.label_index == 1].index

    normal_subset = dataset.subsample(list(normal_indices))
    anomalous_subset = dataset.subsample(list(anomalous_indices))
    return normal_subset, anomalous_subset
