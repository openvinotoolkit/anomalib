"""Unit Tests - MVTec3D Datamodule."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from anomalib import TaskType
from anomalib.data import MVTec3D
from tests.unit.data.base import _TestAnomalibDepthDatamodule


class TestMVTec3D(_TestAnomalibDepthDatamodule):
    """MVTec Datamodule Unit Tests."""

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path, task_type: TaskType) -> MVTec3D:
        """Create and return a Folder 3D datamodule."""
        _datamodule = MVTec3D(
            root=dataset_path / "mvtec_3d",
            category="dummy",
            task=task_type,
            image_size=256,
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
        )
        _datamodule.prepare_data()
        _datamodule.setup()

        return _datamodule

    @pytest.fixture()
    @staticmethod
    def fxt_data_config_path() -> str:
        """Return the path to the test data config."""
        return "configs/data/mvtec_3d.yaml"
