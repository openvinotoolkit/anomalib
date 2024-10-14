"""Unit Tests - Visa Datamodule."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from anomalib import TaskType
from anomalib.data import Visa
from tests.unit.data.base.image import _TestAnomalibImageDatamodule


class TestVisa(_TestAnomalibImageDatamodule):
    """Visa Datamodule Unit Tests."""

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path, task_type: TaskType) -> Visa:
        """Create and return a Avenue datamodule."""
        _datamodule = Visa(
            root=dataset_path,
            category="dummy",
            image_size=256,
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            task=task_type,
        )
        _datamodule.prepare_data()
        _datamodule.setup()

        return _datamodule

    @pytest.fixture()
    @staticmethod
    def fxt_data_config_path() -> str:
        """Return the path to the test data config."""
        return "configs/data/visa.yaml"
