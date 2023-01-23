"""Tests for synthetic anomalous dataset."""

from copy import copy, deepcopy
from pathlib import Path

import pytest

from anomalib.data.folder import FolderDataset
from anomalib.data.synthetic import SyntheticAnomalyDataset
from anomalib.data.utils import get_transforms
from tests.helpers.dataset import get_dataset_path


def get_folder_dataset():
    """Create Folder Dataset."""
    root = get_dataset_path(dataset="bottle")
    transform = get_transforms(image_size=(256, 256))
    dataset = FolderDataset(
        task="segmentation",
        transform=transform,
        root=root,
        normal_dir="good",
        abnormal_dir="broken_large",
        mask_dir="ground_truth/broken_large",
        split="train",
    )
    dataset.setup()

    return dataset


@pytest.fixture(autouse=True)
def make_synthetic_dataset():
    """Create synthetic anomaly dataset from folder dataset."""

    def make():
        folder_dataset = get_folder_dataset()
        synthetic_dataset = SyntheticAnomalyDataset.from_dataset(folder_dataset)
        return synthetic_dataset

    return make


@pytest.fixture(autouse=True)
def synthetic_dataset_from_samples():
    """Create synthetic anomaly dataset by passing a samples dataframe."""
    folder_dataset = get_folder_dataset()
    transform = get_transforms(image_size=(256, 256))
    synthetic_dataset = SyntheticAnomalyDataset(
        task=folder_dataset.task, transform=transform, source_samples=folder_dataset.samples
    )
    return synthetic_dataset


def test_create_synthetic_dataset(make_synthetic_dataset):
    """Tests if the image and mask files listed in the synthetic dataset exist."""
    synthetic_dataset = make_synthetic_dataset()
    assert all(Path(path).exists() for path in synthetic_dataset.samples.image_path)
    assert all(Path(path).exists() for path in synthetic_dataset.samples.mask_path)


def test_create_from_dataset(synthetic_dataset_from_samples):
    """Tests if the image and mask files listed in the synthetic dataset exist, when instantiated from samples df."""
    synthetic_dataset = synthetic_dataset_from_samples
    assert all(Path(path).exists() for path in synthetic_dataset.samples.image_path)
    assert all(Path(path).exists() for path in synthetic_dataset.samples.mask_path)


def test_cleanup(make_synthetic_dataset):
    """Tests if the temporary directory is cleaned up when the instance is deleted."""
    synthetic_dataset = make_synthetic_dataset()
    root = synthetic_dataset.root
    del synthetic_dataset
    assert not root.exists()


def test_copy(make_synthetic_dataset):
    """Tests if the dataset is copied correctly, and files still exist after original instance is deleted."""
    synthetic_dataset = make_synthetic_dataset()
    synthetic_dataset_cp = copy(synthetic_dataset)
    assert all(synthetic_dataset.samples == synthetic_dataset_cp.samples)
    del synthetic_dataset
    assert synthetic_dataset_cp.root.exists()


def test_deepcopy(make_synthetic_dataset):
    """Tests if the dataset is deep-copied correctly, and files still exist after original instance is deleted."""
    synthetic_dataset = make_synthetic_dataset()
    synthetic_dataset_cp = deepcopy(synthetic_dataset)
    assert all(synthetic_dataset.samples == synthetic_dataset_cp.samples)
    del synthetic_dataset
    assert synthetic_dataset_cp.root.exists()
