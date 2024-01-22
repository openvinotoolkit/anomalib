"""Unit Tests - MVTec Datamodule."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from anomalib import TaskType
from anomalib.data import MVTec
from tests.unit.data.base.image import _TestAnomalibImageDatamodule


class TestMVTec(_TestAnomalibImageDatamodule):
    """MVTec Datamodule Unit Tests."""

    @pytest.fixture()
    def datamodule(self, dataset_path: Path, task_type: TaskType) -> MVTec:
        """Create and return a MVTec datamodule."""
        _datamodule = MVTec(
            root=dataset_path / "mvtec",
            category="dummy",
            task=task_type,
            image_size=256,
            train_batch_size=4,
            eval_batch_size=4,
        )
        _datamodule.prepare_data()
        _datamodule.setup()

        return _datamodule
