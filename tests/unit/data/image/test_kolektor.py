"""Unit Tests - Kolektor Datamodule."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from anomalib import TaskType
from anomalib.data import Kolektor
from tests.unit.data.base.image import _TestAnomalibImageDatamodule


class TestKolektor(_TestAnomalibImageDatamodule):
    """Kolektor Datamodule Unit Tests."""

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path, task_type: TaskType) -> Kolektor:
        """Create and return a BTech datamodule."""
        _datamodule = Kolektor(
            root=dataset_path / "kolektor",
            task=task_type,
            image_size=256,
            train_batch_size=4,
            eval_batch_size=4,
        )

        _datamodule.prepare_data()
        _datamodule.setup()

        return _datamodule

    @pytest.fixture()
    @staticmethod
    def fxt_data_config_path() -> str:
        """Return the path to the test data config."""
        return "configs/data/kolektor.yaml"
