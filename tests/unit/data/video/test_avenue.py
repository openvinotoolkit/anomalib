"""Unit Tests - Avenue Datamodule."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from anomalib import TaskType
from anomalib.data import Avenue
from tests.unit.data.base.video import _TestAnomalibVideoDatamodule


class TestAvenue(_TestAnomalibVideoDatamodule):
    """Avenue Datamodule Unit Tests."""

    @pytest.fixture()
    def datamodule(self, dataset_path: Path, task_type: TaskType) -> Avenue:
        """Create and return a Avenue datamodule."""
        _datamodule = Avenue(
            root=dataset_path / "avenue",
            gt_dir=dataset_path / "avenue" / "ground_truth_demo",
            image_size=256,
            task=task_type,
            num_workers=0,
            train_batch_size=4,
            eval_batch_size=4,
        )

        _datamodule.prepare_data()
        _datamodule.setup()

        return _datamodule
