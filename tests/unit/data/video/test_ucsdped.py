"""Unit Tests - UCSDped Datamodule."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from anomalib import TaskType, UCSDped
from tests.unit.data.base.video import _TestAnomalibVideoDatamodule


class TestUCSDped(_TestAnomalibVideoDatamodule):
    """UCSDped Datamodule Unit Tests."""

    @pytest.fixture()
    def datamodule(self, dataset_path: Path, task_type: TaskType) -> UCSDped:
        """Create and return a UCSDped datamodule."""
        _datamodule = UCSDped(
            root=dataset_path / "ucsdped",
            category="dummy",
            clip_length_in_frames=1,
            frames_between_clips=1,
            task=task_type,
            image_size=256,
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
        )
        _datamodule.prepare_data()
        _datamodule.setup()

        return _datamodule
