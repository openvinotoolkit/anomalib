"""Unit Tests - BTech Datamodule."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from anomalib import TaskType
from anomalib.data import BTech
from tests.unit.data.base.image import _TestAnomalibImageDatamodule


class TestBTech(_TestAnomalibImageDatamodule):
    """MVTec Datamodule Unit Tests."""

    @pytest.fixture()
    def datamodule(self, dataset_path: Path, task_type: TaskType) -> BTech:
        """Create and return a BTech datamodule."""
        _datamodule = BTech(
            root=dataset_path / "btech",
            category="dummy",
            task=task_type,
            image_size=256,
            train_batch_size=4,
            eval_batch_size=4,
        )

        _datamodule.prepare_data()
        _datamodule.setup()

        return _datamodule
