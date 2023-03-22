"""Unit Tests - BTech Datamodule."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from anomalib.data import BTech, TaskType
from tests.helpers.data import get_dataset_path

from .base import _TestAnomalibImageDatamodule


class TestBTech(_TestAnomalibImageDatamodule):
    """MVTec Datamodule Unit Tests."""

    @pytest.fixture
    def datamodule(self, task_type: TaskType) -> BTech:
        # Create and prepare the dataset
        _datamodule = BTech(
            root=get_dataset_path("BTech"),
            category="01",
            task=task_type,
            image_size=256,
            train_batch_size=4,
            eval_batch_size=4,
        )

        _datamodule.prepare_data()
        _datamodule.setup()

        return _datamodule
