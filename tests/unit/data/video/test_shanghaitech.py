"""Unit Tests - ShanghaiTech Datamodule."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from anomalib import TaskType
from anomalib.data import ShanghaiTech
from tests.unit.data.base.video import _TestAnomalibVideoDatamodule


class TestShanghaiTech(_TestAnomalibVideoDatamodule):
    """ShanghaiTech Datamodule Unit Tests."""

    @pytest.fixture()
    @staticmethod
    def clip_length_in_frames() -> int:
        """Return the number of frames in each clip."""
        return 2

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path, task_type: TaskType, clip_length_in_frames: int) -> ShanghaiTech:
        """Create and return a Shanghai datamodule."""
        _datamodule = ShanghaiTech(
            root=dataset_path / "shanghaitech",
            scene=1,
            clip_length_in_frames=clip_length_in_frames,
            image_size=(256, 256),
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            task=task_type,
        )

        _datamodule.prepare_data()
        _datamodule.setup()

        return _datamodule

    @pytest.fixture()
    @staticmethod
    def fxt_data_config_path() -> str:
        """Return the path to the test data config."""
        return "configs/data/shanghaitech.yaml"
