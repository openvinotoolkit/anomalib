"""Tests for synthetic anomalous dataset."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import copy, deepcopy
from pathlib import Path

import pytest

from anomalib.data import AnomalibDataset, TaskType
from anomalib.data.folder import FolderDataset
from anomalib.data.synthetic import SyntheticAnomalyDataset
from anomalib.data.utils import get_transforms, random_split
from tests.helpers.data import get_dataset_path


@pytest.fixture(autouse=True)
def sample_dataset() -> AnomalibDataset:
    """Creates a folder dataset that is subsampled for faster testing.

    Returns:
        AnomalibDataset: Subsampled folder dataset.
    """
    # Create a folder dataset.
    dataset = FolderDataset(
        root=get_dataset_path(dataset="bottle"),
        normal_dir="good",
        abnormal_dir="broken_large",
        mask_dir="ground_truth/broken_large",
        split="train",
        task=TaskType.SEGMENTATION,
        transform=get_transforms(image_size=256),
    )
    dataset.setup()

    # Subsample the dataset to make it faster.
    _, _sample_dataset = random_split(dataset, split_ratio=0.01, label_aware=True)

    return _sample_dataset


@pytest.fixture(autouse=True)
def synthetic_dataset(sample_dataset: AnomalibDataset) -> SyntheticAnomalyDataset:
    """Creates a synthetic dataset from a folder dataset.

    Args:
        sample_dataset (AnomalibDataset): Sampled folder dataset.

    Returns:
        SyntheticAnomalyDataset: Synthetic dataset.
    """
    return SyntheticAnomalyDataset.from_dataset(sample_dataset)


@pytest.fixture(autouse=True)
def synthetic_dataset_from_samples(sample_dataset: AnomalibDataset) -> SyntheticAnomalyDataset:
    """Creates a synthetic dataset from a folder dataset.

    Args:
        sample_dataset (AnomalibDataset): Sampled folder dataset.

    Returns:
        SyntheticAnomalyDataset: Synthetic dataset.
    """
    _synthetic_dataset = SyntheticAnomalyDataset(
        task=sample_dataset.task,
        transform=get_transforms(image_size=256),
        source_samples=sample_dataset.samples,
    )
    return _synthetic_dataset


def test_create_synthetic_dataset(synthetic_dataset: SyntheticAnomalyDataset) -> None:
    """Tests if the image and mask files listed in the synthetic dataset exist."""
    assert all(Path(path).exists() for path in synthetic_dataset.samples.image_path)
    assert all(Path(path).exists() for path in synthetic_dataset.samples.mask_path)


def test_create_from_dataset(synthetic_dataset_from_samples: SyntheticAnomalyDataset) -> None:
    """Tests if the image and mask files listed in the synthetic dataset exist, when instantiated from samples df."""
    synthetic_dataset = synthetic_dataset_from_samples
    assert all(Path(path).exists() for path in synthetic_dataset.samples.image_path)
    assert all(Path(path).exists() for path in synthetic_dataset.samples.mask_path)


def test_cleanup(synthetic_dataset: SyntheticAnomalyDataset) -> None:
    """Tests if the temporary directory is cleaned up when the instance is deleted."""
    synthetic_dataset_cp = copy(synthetic_dataset)
    root = synthetic_dataset_cp.root
    del synthetic_dataset_cp
    assert not root.exists()


def test_copy(synthetic_dataset: SyntheticAnomalyDataset) -> None:
    """Tests if the dataset is copied correctly, and files still exist after original instance is deleted."""
    synthetic_dataset_cp = copy(synthetic_dataset)
    assert all(synthetic_dataset.samples == synthetic_dataset_cp.samples)
    del synthetic_dataset
    assert synthetic_dataset_cp.root.exists()


def test_deepcopy(synthetic_dataset: SyntheticAnomalyDataset) -> None:
    """Tests if the dataset is deep-copied correctly, and files still exist after original instance is deleted."""
    synthetic_dataset_cp = deepcopy(synthetic_dataset)
    assert all(synthetic_dataset.samples == synthetic_dataset_cp.samples)
    del synthetic_dataset
    assert synthetic_dataset_cp.root.exists()
