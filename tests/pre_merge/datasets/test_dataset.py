"""Test the AnomalibDataset class."""

import random

import pandas as pd
import pytest

from anomalib.data import TaskType
from anomalib.data.folder import FolderDataset
from anomalib.data.utils import get_transforms
from anomalib.data.utils.split import concatenate_datasets, random_split
from tests.helpers.dataset import get_dataset_path


@pytest.fixture(autouse=True)
def folder_dataset():
    """Create Folder Dataset."""
    root = get_dataset_path(dataset="bottle")
    transform = get_transforms(image_size=(256, 256))
    dataset = FolderDataset(
        task=TaskType.CLASSIFICATION,
        transform=transform,
        root=root,
        normal_dir="good",
        abnormal_dir="broken_large",
    )
    dataset.setup()

    return dataset


class TestAnomalibDataset:
    def test_subsample(self, folder_dataset):
        """Test the subsample functionality."""

        sample_size = int(0.5 * len(folder_dataset))
        indices = random.sample(range(len(folder_dataset)), sample_size)
        subset = folder_dataset.subsample(indices)

        # check if the dataset has been subsampled to correct size
        assert len(subset) == sample_size
        # check if index has been reset
        assert subset.samples.index.start == 0
        assert subset.samples.index.stop == sample_size

    def test_random_split(self, folder_dataset):
        """Test the random subset splitting."""

        # subset splitting
        subsets = random_split(folder_dataset, [0.4, 0.35, 0.25])
        # check if subset splitting has been performed correctly
        assert len(subsets) == 3
        # reconstruct the original dataset by concatenating the subsets
        reconstructed_dataset = concatenate_datasets(subsets)
        # check if reconstructed dataset is equal to original dataset
        assert folder_dataset.samples.equals(reconstructed_dataset.samples)

        # check if warning raised when one of the subsets is empty
        split_ratios = [1 - (1 / (len(folder_dataset) + 1)), 1 / (len(folder_dataset) + 1)]
        with pytest.warns():
            subsets = random_split(folder_dataset, split_ratios)

        # label-aware subset splitting
        samples = folder_dataset.samples
        normal_samples = samples[samples["label_index"] == 0]
        anomalous_samples = samples[samples["label_index"] == 1]
        samples = pd.concat([normal_samples, anomalous_samples[0:5]])
        folder_dataset.samples = samples

        subsets = random_split(folder_dataset, [0.4, 0.4, 0.2], label_aware=True)

        # 5 anomalous images in total, so the first two subsets should each have 2, and the last subset 1
        assert len(subsets[0].samples[subsets[0].samples["label_index"] == 1]) == 2
        assert len(subsets[1].samples[subsets[1].samples["label_index"] == 1]) == 2
        assert len(subsets[2].samples[subsets[2].samples["label_index"] == 1]) == 1
