"""Unit Tests - Visa Datamodule."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from anomalib.data import TaskType, Visa
from tests.helpers.data import get_dataset_path

from .base import _TestAnomalibImageDatamodule


class TestVisa(_TestAnomalibImageDatamodule):
    """Visa Datamodule Unit Tests."""

    @pytest.fixture
    def datamodule(self, task_type: TaskType) -> Visa:
        # Create and prepare the dataset
        _datamodule = Visa(
            root=get_dataset_path("visa"),
            category="candle",
            image_size=256,
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            task=task_type,
        )
        _datamodule.prepare_data()
        _datamodule.setup()

        return _datamodule
