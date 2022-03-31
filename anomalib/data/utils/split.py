"""Dataset Split Utils.

This module contains function in regards to splitting normal images in training set,
and creating validation sets from test sets.

These function are useful
    - when the test set does not contain any normal images.
    - when the dataset doesn't have a validation set.
"""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import random

from pandas.core.frame import DataFrame


def split_normal_images_in_train_set(
    samples: DataFrame, split_ratio: float = 0.1, seed: int = 0, normal_label: str = "good"
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

    if seed > 0:
        random.seed(seed)

    normal_train_image_indices = samples.index[(samples.split == "train") & (samples.label == normal_label)].to_list()
    num_normal_train_images = len(normal_train_image_indices)
    num_normal_valid_images = int(num_normal_train_images * split_ratio)

    indices_to_split_from_train_set = random.sample(population=normal_train_image_indices, k=num_normal_valid_images)
    samples.loc[indices_to_split_from_train_set, "split"] = "test"

    return samples


def create_validation_set_from_test_set(samples: DataFrame, seed: int = 0, normal_label: str = "good") -> DataFrame:
    """Craete Validation Set from Test Set.

    This function creates a validation set from test set by splitting both
    normal and abnormal samples to two.

    Args:
        samples (DataFrame): Dataframe containing dataset info such as filenames, splits etc.
        seed (int, optional): Random seed to ensure reproducibility. Defaults to 0.
        normal_label (str): Name of the normal label. For MVTec AD, for instance, this is normal_label.
    """

    if seed > 0:
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
