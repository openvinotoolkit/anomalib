"""Unit Tests - CSV Datamodule."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from anomalib import TaskType
from anomalib.data import CSV
from tests.unit.data.base.image import _TestAnomalibImageDatamodule


class TestCSV(_TestAnomalibImageDatamodule):
    """CSV Datamodule Unit Tests.

    All of the folder datamodule tests are placed in ``TestCSV`` class.
    """

    @pytest.fixture()
    def datamodule(self, dataset_path: Path, task_type: TaskType) -> CSV:
        """Create and return a CSV datamodule."""
        # Create and prepare the dataset
        _datamodule = CSV(
            name="dummy_csv",
            csv_path=dataset_path / "csv" / "samples.csv",
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            task=task_type,
            test_split_mode="auto",
            val_split_mode="auto",
        )
        _datamodule.setup()

        return _datamodule

    @pytest.fixture()
    def fxt_data_config_path(self) -> str:
        """Return the path to the test data config."""
        return "configs/data/folder.yaml"
