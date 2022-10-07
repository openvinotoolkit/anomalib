"""Dataset Split Utils.

This module contains function in regards to splitting normal images in training set,
and creating validation sets from test sets.

These function are useful
    - when the test set does not contain any normal images.
    - when the dataset doesn't have a validation set.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import random
from typing import Optional, Tuple

from pandas.core.frame import DataFrame
from torch.utils.data import Subset

from anomalib.data.base import AnomalibDataset


def split_normal_images_in_train_set(
    samples: DataFrame, split_ratio: float = 0.1, seed: Optional[int] = None, normal_label: str = "good"
) -> DataFrame:
    """Split normal images in train set.

        This function splits the normal images in training set and assigns the
        values to the test set. This is particularly useful especially when the
        test set does not contain any normal images.

        This is important because when the test set doesn't have any normal images,
        AUC computation fails due to having single class.

    Args:
        samples (DataFrame): Dataframe containing dataset info such as filenames, splits etc.
        split_ratio (float, optional): Train-Test normal image split ratio. Defaults to 0.1.
        seed (int, optional): Random seed to ensure reproducibility. Defaults to 0.
        normal_label (str): Name of the normal label. For MVTec AD, for instance, this is normal_label.

    Returns:
        DataFrame: Output dataframe where the part of the training set is assigned to test set.
    """

    if seed is not None:
        random.seed(seed)

    normal_train_image_indices = samples.index[(samples.split == "train") & (samples.label == normal_label)].to_list()
    num_normal_train_images = len(normal_train_image_indices)
    num_normal_valid_images = int(num_normal_train_images * split_ratio)

    indices_to_split_from_train_set = random.sample(population=normal_train_image_indices, k=num_normal_valid_images)
    samples.loc[indices_to_split_from_train_set, "split"] = "test"

    return samples


def create_validation_set_from_test_set(
    samples: DataFrame, seed: Optional[int] = None, normal_label: str = "good"
) -> DataFrame:
    """Craete Validation Set from Test Set.

    This function creates a validation set from test set by splitting both
    normal and abnormal samples to two.

    Args:
        samples (DataFrame): Dataframe containing dataset info such as filenames, splits etc.
        seed (int, optional): Random seed to ensure reproducibility. Defaults to 0.
        normal_label (str): Name of the normal label. For MVTec AD, for instance, this is normal_label.
    """

    if seed is not None:
        random.seed(seed)

    # Split normal images.
    normal_test_image_indices = samples.index[(samples.split == "test") & (samples.label == normal_label)].to_list()
    num_normal_valid_images = len(normal_test_image_indices) // 2

    indices_to_sample = random.sample(population=normal_test_image_indices, k=num_normal_valid_images)
    samples.loc[indices_to_sample, "split"] = "val"

    # Split abnormal images.
    abnormal_test_image_indices = samples.index[(samples.split == "test") & (samples.label != normal_label)].to_list()
    num_abnormal_valid_images = len(abnormal_test_image_indices) // 2

    indices_to_sample = random.sample(population=abnormal_test_image_indices, k=num_abnormal_valid_images)
    samples.loc[indices_to_sample, "split"] = "val"

    return samples


def split_normals_and_anomalous(
    dataset: "AnomalibDataset", split_ratio: float, seed: Optional[int] = None
) -> Tuple[Subset, Subset]:
    """Wrap dataset wit torch.utils.data.Subset twice to create two (non-overlaping) subsets.
    Args:
        dataset (AnomalibDataset): AnomalibDataset object.
        split_ratio (float): Split ratio (0 to 100%) that goes to the NEW split.
        seed (int): Random seed to ensure reproducibility.
    Returns:
        Tuple[AnomalibDataset, AnomalibDataset]: (new split, old split).
    """

    assert 0 < split_ratio < 1, "Split ratio must be between 0 and 1."
    if seed is not None:
        assert seed >= 0, "Seed must be non-negative."
        random.seed(seed)

    # get the indices of the normal/anomalous images in the dataset
    normals_indices = dataset.samples.index[dataset.samples.label_index == 0].to_list()
    anomalous_indices = dataset.samples.index[dataset.samples.label_index == 1].to_list()

    # get the number of normal/anomalous images will got to the new split
    new_split_n_normals = int(len(normals_indices) * split_ratio)
    new_split_n_anomalous = int(len(anomalous_indices) * split_ratio)

    # ranmdomly sample the indices of the normal/anomalous images that will go to the new split
    new_split_normals_indices = random.sample(population=normals_indices, k=new_split_n_normals)
    new_split_anomalous_indices = random.sample(population=anomalous_indices, k=new_split_n_anomalous)

    # indices that remain in the original split
    old_split_normals_indices = list(set(normals_indices) - set(new_split_normals_indices))
    old_split_anomalous_indices = list(set(anomalous_indices) - set(new_split_anomalous_indices))

    # create the new split and the (reduced) original split
    # new_split = Subset(dataset, new_split_normals_indices + new_split_anomalous_indices)
    # old_split = Subset(dataset, old_split_normals_indices + old_split_anomalous_indices)
    new_split = dataset.subsample(new_split_normals_indices + new_split_anomalous_indices)
    old_split = dataset.subsample(old_split_normals_indices + old_split_anomalous_indices)

    return new_split, old_split
