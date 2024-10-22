"""Unit tests - Datumaro Datamodule."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from anomalib import TaskType
from anomalib.data import Datumaro
from tests.unit.data.base.image import _TestAnomalibImageDatamodule


class TestDatumaro(_TestAnomalibImageDatamodule):
    """Datumaro Datamodule Unit Tests."""

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path, task_type: TaskType) -> Datumaro:
        """Create and return a Datumaro datamodule."""
        if task_type != TaskType.CLASSIFICATION:
            pytest.skip("Datumaro only supports classification tasks.")

        _datamodule = Datumaro(
            root=dataset_path / "datumaro",
            task=task_type,
            train_batch_size=4,
            eval_batch_size=4,
        )
        _datamodule.setup()

        return _datamodule

    @pytest.fixture()
    @staticmethod
    def fxt_data_config_path() -> str:
        """Return the path to the test data config."""
        return "configs/data/datumaro.yaml"
