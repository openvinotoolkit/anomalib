"""Test the Video dataset class and utils."""

from pathlib import Path

import pytest

from anomalib.data import TaskType
from anomalib.data.ucsd_ped import (
    UCSDpedClipsIndexer,
    UCSDpedDataset,
    make_ucsd_dataset,
)
from anomalib.data.utils import get_transforms
from anomalib.data.utils.split import Split, random_split
from tests.helpers.dataset import get_dataset_path


@pytest.fixture
def ucsd_clips(n_frames, stride, split=Split.TEST):
    """Create Folder Dataset."""
    root = get_dataset_path(dataset="ucsd")
    path = Path(root) / "UCSDped2"
    samples = make_ucsd_dataset(path=path, split=split)
    clips = UCSDpedClipsIndexer(
        video_paths=samples.image_path,
        mask_paths=samples.mask_path,
        clip_length_in_frames=n_frames,
        frames_between_clips=stride,
    )
    return clips


@pytest.fixture
def ucsd_dataset(split):
    root = get_dataset_path(dataset="ucsd")
    transform = get_transforms(image_size=(256, 256))
    dataset = UCSDpedDataset(
        task=TaskType.CLASSIFICATION,
        root=Path(root),
        category="UCSDped2",
        transform=transform,
        clip_length_in_frames=16,
        frames_between_clips=1,
        split=split,
    )
    dataset.setup()
    return dataset


class TestClipsIndexer:
    @pytest.mark.parametrize("n_frames", [1, 8, 16])
    @pytest.mark.parametrize("stride", [1, 8, 16])
    def test_clip_length_and_stride(self, ucsd_clips, n_frames, stride):
        # check clip length
        clip = ucsd_clips.get_item(0)
        assert clip["image"].shape[0] == n_frames
        assert clip["mask"].shape[0] == n_frames
        # check frames between clips
        next_clip = ucsd_clips.get_item(1)
        assert next_clip["frames"][0] - clip["frames"][0] == stride


class TestVideoDataset:
    @pytest.mark.parametrize("split", [Split.TRAIN, Split.TEST])
    def test_indexing(self, ucsd_dataset):
        # check if clips are indexed at setup
        assert isinstance(ucsd_dataset.indexer, UCSDpedClipsIndexer)
        # check if clips are re-indexed after splitting
        split0, split1 = random_split(ucsd_dataset, 0.5)
        assert len(split0) + len(split1) == len(ucsd_dataset)

    @pytest.mark.parametrize(
        "split, required_keys",
        [
            (Split.TRAIN, ("video_path", "frames", "last_frame", "image", "original_image")),
            (Split.TEST, ("video_path", "frames", "label", "last_frame", "image", "original_image", "mask")),
        ],
    )
    def test_get_item(self, ucsd_dataset, required_keys):
        item = next(iter(ucsd_dataset))
        # confirm that all the required keys are there
        assert set(item.keys()) == set(required_keys)
