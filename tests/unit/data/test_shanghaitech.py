"""Unit Tests - ShanghaiTech Datamodule."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import random

import pytest

from anomalib.data import ShanghaiTech, TaskType
from anomalib.data.utils.split import ValSplitMode

from .base import _TestAnomalibVideoDatamodule


class TestShanghaiTech(_TestAnomalibVideoDatamodule):
    """ShanghaiTech Datamodule Unit Tests."""

    @pytest.fixture
    def datamodule(self, task_type: TaskType) -> ShanghaiTech:
        # Create and prepare the dataset
        _datamodule = ShanghaiTech(
            root="./datasets/shanghaitech",
            scene=random.randint(1, 13),
            clip_length_in_frames=1,
            frames_between_clips=1,
            task=task_type,
            image_size=256,
            center_crop=256,
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            # NOTE: ``ValSplitMode.FROM_TEST`` returns an error.
            val_split_mode=ValSplitMode.SAME_AS_TEST,
        )

        _datamodule.prepare_data()
        _datamodule.setup()

        return _datamodule
