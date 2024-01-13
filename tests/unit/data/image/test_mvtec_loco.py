"""Unit Tests - MVTecLoco Datamodule."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from anomalib import TaskType
from anomalib.data import MVTecLoco
from tests.unit.data.base.image import _TestAnomalibImageDatamodule


class TestMVTecLoco(_TestAnomalibImageDatamodule):
    """MVTecLoco Datamodule Unit Tests."""

    @pytest.fixture()
    def datamodule(self, dataset_path: Path, task_type: TaskType) -> MVTecLoco:
        """Create and return a MVTecLoco datamodule."""
        _datamodule = MVTecLoco(
            root=dataset_path / "mvtec_loco",
            category="dummy",
            task=task_type,
            image_size=256,
            train_batch_size=4,
            eval_batch_size=4,
        )
        _datamodule.prepare_data()
        _datamodule.setup()

        return _datamodule
