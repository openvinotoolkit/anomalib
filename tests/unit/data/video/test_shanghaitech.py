"""Unit Tests - ShanghaiTech Datamodule."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import pytest

from anomalib import TaskType
from anomalib.data import ShanghaiTech
from tests.unit.data.base.video import _TestAnomalibVideoDatamodule


class TestShanghaiTech(_TestAnomalibVideoDatamodule):
    """ShanghaiTech Datamodule Unit Tests."""

    @pytest.fixture()
    def datamodule(self, dataset_path: Path, task_type: TaskType) -> ShanghaiTech:
        """Create and return a Shanghai datamodule."""
        _datamodule = ShanghaiTech(
            root=dataset_path / "shanghaitech",
            scene=1,
            image_size=(256, 256),
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            task=task_type,
        )

        _datamodule.prepare_data()
        _datamodule.setup()

        return _datamodule
