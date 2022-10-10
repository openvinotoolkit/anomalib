"""Dataset Split Utils.

This module contains function in regards to splitting normal images in training set,
and creating validation sets from test sets.

These function are useful
    - when the test set does not contain any normal images.
    - when the dataset doesn't have a validation set.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math
import warnings
from typing import Sequence, Union

from torch import randperm, split

from anomalib.data.base import AnomalibDataset


def random_split(
    dataset: AnomalibDataset, split_ratio: Union[float, Sequence[float]], label_aware: bool = False
) -> Sequence[AnomalibDataset]:
    """Perform a random split of a dataset.

    Args:
        dataset (AnomalibDataset): Source dataset
        split_ratio (Union[float, Sequence[float]]): Fractions of the splits that will be produced. The values in the
            sequence must sum to 1. If a single value is passed, the ratio will be converted to
            [1-split_ratio, split_ratio].
        label_aware (bool): When True, the relative occurrence of the different class labels of the source dataset will
            be maintained in each of the subsets.
    """

    if isinstance(split_ratio, float):
        split_ratio = [1 - split_ratio, split_ratio]

    assert math.isclose(sum(split_ratio), 1) and sum(split_ratio) <= 1, "split ratios must sum to 1."
    assert all(0 < ratio < 1 for ratio in split_ratio), "all split ratios must be between 0 and 1."

    # create list of source data
    if label_aware:
        indices_per_label = [group.index for _, group in dataset.samples.groupby("label_index")]
        datasets = [dataset.subsample(indices) for indices in indices_per_label]
    else:
        datasets = [dataset]

    # split each (label-aware) subset of source data
    subsets = []
    for dataset in datasets:
        # get subset lengths
        subset_lengths = []
        for ratio in split_ratio:
            subset_lengths.append(int(math.floor(len(dataset) * ratio)))
        for i in range(len(dataset) - sum(subset_lengths)):
            subset_idx = i % sum(subset_lengths)
            subset_lengths[subset_idx] += 1
        for index, length in enumerate(subset_lengths):
            if length == 0:
                warnings.warn(f"Length of subset at index {index} is 0.")
        # perform random subsampling
        indices = randperm(len(dataset))
        subsets.append([dataset.subsample(subset_indices) for subset_indices in split(indices, subset_lengths)])

    # concatenate and return
    subsets = list(map(list, zip(*subsets)))
    return tuple(sum(subset) for subset in subsets)
