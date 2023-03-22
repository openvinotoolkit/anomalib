"""Unit Tests - ShanghaiTech Datamodule."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from anomalib.data import ShanghaiTech, TaskType
from anomalib.data.utils.split import ValSplitMode
from tests.helpers.dataset import get_dataset_path

from .base import _TestAnomalibVideoDatamodule


class TestShanghaiTech(_TestAnomalibVideoDatamodule):
    """ShanghaiTech Datamodule Unit Tests."""

    @pytest.fixture
    def datamodule(self, task_type: TaskType) -> ShanghaiTech:
        # Create and prepare the dataset
        _datamodule = ShanghaiTech(
            root=get_dataset_path("shanghaitech"),
            scene=1,
            image_size=(256, 256),
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            task=task_type,
            val_split_mode=ValSplitMode.FROM_TEST,
        )

        _datamodule.prepare_data()
        _datamodule.setup()

        return _datamodule
