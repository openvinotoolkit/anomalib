"""Dataset Split Utils.

This module contains function in regards to splitting normal images in training set,
and creating validation sets from test sets.

These function are useful
    - when the test set does not contain any normal images.
    - when the dataset doesn't have a validation set.
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from anomalib import data

logger = logging.getLogger(__name__)


class Split(str, Enum):
    """Split of a subset."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class TestSplitMode(str, Enum):
    """Splitting mode used to obtain subset."""

    NONE = "none"
    FROM_DIR = "from_dir"
    SYNTHETIC = "synthetic"


class ValSplitMode(str, Enum):
    """Splitting mode used to obtain validation subset."""

    NONE = "none"
    SAME_AS_TEST = "same_as_test"
    FROM_TRAIN = "from_train"
    FROM_TEST = "from_test"
    SYNTHETIC = "synthetic"


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


def split_by_label(dataset: "data.AnomalibDataset") -> tuple["data.AnomalibDataset", "data.AnomalibDataset"]:
    """Split the dataset into the normal and anomalous subsets."""
    samples = dataset.samples
    normal_indices = samples[samples.label_index == 0].index
    anomalous_indices = samples[samples.label_index == 1].index

    normal_subset = dataset.subsample(list(normal_indices))
    anomalous_subset = dataset.subsample(list(anomalous_indices))
    return normal_subset, anomalous_subset
