"""Unit Tests - MVTec Datamodule."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from anomalib.data import MVTec, TaskType
import pytest
from .base import _TestAnomalibImageDatamodule


class TestMVTec(_TestAnomalibImageDatamodule):
    """MVTec Datamodule Unit Tests."""

    @pytest.fixture
    def datamodule(self, task_type: TaskType) -> MVTec:
        # Create and prepare the dataset
        _datamodule = MVTec(
            root="./datasets/MVTec",
            category="bottle",
            task=task_type,
            image_size=256,
            train_batch_size=4,
            eval_batch_size=4,
        )
        _datamodule.prepare_data()
        _datamodule.setup()

        return _datamodule
