"""Tests for synthetic anomalous dataset."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import copy
from pathlib import Path

import pytest

from anomalib import TaskType
from anomalib.data.image.folder import FolderDataset
from anomalib.data.utils.synthetic import SyntheticAnomalyDataset


@pytest.fixture(scope="module")
def folder_dataset(dataset_path: Path) -> FolderDataset:
    """Fixture that returns a FolderDataset instance."""
    return FolderDataset(
        name="dummy",
        task=TaskType.SEGMENTATION,
        root=dataset_path / "mvtec" / "dummy",
        normal_dir="train/good",
        abnormal_dir="test/bad",
        normal_test_dir="test/good",
        mask_dir="ground_truth/bad",
        split="train",
    )


@pytest.fixture(scope="module")
def synthetic_dataset(folder_dataset: FolderDataset) -> SyntheticAnomalyDataset:
    """Fixture that returns a SyntheticAnomalyDataset instance."""
    return SyntheticAnomalyDataset.from_dataset(folder_dataset)


@pytest.fixture(scope="module")
def synthetic_dataset_from_samples(folder_dataset: FolderDataset) -> SyntheticAnomalyDataset:
    """Fixture that returns a SyntheticAnomalyDataset instance."""
    return SyntheticAnomalyDataset(
        task=folder_dataset.task,
        transform=folder_dataset.transform,
        source_samples=folder_dataset.samples,
    )


class TestSyntheticAnomalyDataset:
    """Test SyntheticAnomalyDataset class."""

    @staticmethod
    def test_create_synthetic_dataset(synthetic_dataset: SyntheticAnomalyDataset) -> None:
        """Tests if the image and mask files listed in the synthetic dataset exist."""
        assert all(Path(path).exists() for path in synthetic_dataset.samples.image_path)
        assert all(Path(path).exists() for path in synthetic_dataset.samples.mask_path)

    @staticmethod
    def test_create_from_dataset(synthetic_dataset_from_samples: SyntheticAnomalyDataset) -> None:
        """Test if the synthetic dataset is instantiated correctly from samples df."""
        assert all(Path(path).exists() for path in synthetic_dataset_from_samples.samples.image_path)
        assert all(Path(path).exists() for path in synthetic_dataset_from_samples.samples.mask_path)

    @staticmethod
    def test_copy(synthetic_dataset: SyntheticAnomalyDataset) -> None:
        """Tests if the dataset is copied correctly, and files still exist after original instance is deleted."""
        synthetic_dataset_cp = copy(synthetic_dataset)
        assert all(synthetic_dataset.samples == synthetic_dataset_cp.samples)
        del synthetic_dataset
        assert synthetic_dataset_cp.root.exists()

    @staticmethod
    def test_cleanup(folder_dataset: FolderDataset) -> None:
        """Tests if the temporary directory is cleaned up when the instance is deleted."""
        synthetic_dataset = SyntheticAnomalyDataset.from_dataset(folder_dataset)
        root = synthetic_dataset.root
        del synthetic_dataset
        assert not root.exists()
