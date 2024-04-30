"""Unit Tests - Folder Datamodule."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from anomalib import TaskType
from anomalib.data import FolderVideo
from tests.unit.data.base.video import _TestAnomalibVideoDatamodule


class TestFolderVideo(_TestAnomalibVideoDatamodule):
    """FolderVideo Datamodule Unit Tests.

    All of the folder datamodule tests are placed in ``TestFolder`` class.
    """

    @pytest.fixture()
    def clip_length_in_frames(self) -> int:
        """Return the number of frames in each clip."""
        return 2

    @pytest.fixture()
    def datamodule(self, dataset_path: Path, task_type: TaskType, clip_length_in_frames: int) -> FolderVideo:
        """Create and return a FolderVideo datamodule."""
        _datamodule = FolderVideo(
            root=dataset_path / "foldervideo",
            normal_dir="train_vid",
            test_dir="test_vid",
            mask_dir="test_labels",
            clip_length_in_frames=clip_length_in_frames,
            task=task_type,
            image_size=256,
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
        )
        _datamodule.prepare_data()
        _datamodule.setup()

        return _datamodule
