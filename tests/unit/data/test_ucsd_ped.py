"""Unit Tests - UCSDped Datamodule."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from anomalib.data import UCSDped, TaskType
import pytest
from .base import _TestAnomalibVideoDatamodule


class TestUCSDped(_TestAnomalibVideoDatamodule):
    """UCSDped Datamodule Unit Tests."""

    @pytest.fixture
    def datamodule(self, task_type: TaskType) -> UCSDped:
        # Create and prepare the dataset
        _datamodule = UCSDped(
            root="./datasets/ucsd",
            category="UCSDped2",
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
