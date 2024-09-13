"""Unit Tests - Avenue Datamodule."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from anomalib import TaskType
from anomalib.data import Avenue
from tests.unit.data.base.video import _TestAnomalibVideoDatamodule


class TestAvenue(_TestAnomalibVideoDatamodule):
    """Avenue Datamodule Unit Tests."""

    @pytest.fixture()
    @staticmethod
    def clip_length_in_frames() -> int:
        """Return the number of frames in each clip."""
        return 2

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path, task_type: TaskType, clip_length_in_frames: int) -> Avenue:
        """Create and return a Avenue datamodule."""
        _datamodule = Avenue(
            root=dataset_path / "avenue",
            gt_dir=dataset_path / "avenue" / "ground_truth_demo",
            clip_length_in_frames=clip_length_in_frames,
            image_size=256,
            task=task_type,
            num_workers=0,
            train_batch_size=4,
            eval_batch_size=4,
        )

        _datamodule.prepare_data()
        _datamodule.setup()

        return _datamodule

    @pytest.fixture()
    @staticmethod
    def fxt_data_config_path() -> str:
        """Return the path to the test data config."""
        return "configs/data/avenue.yaml"
