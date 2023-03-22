"""Unit Tests - MVTec3D Datamodule."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from anomalib.data import MVTec3D, TaskType
from tests.helpers.dataset import get_dataset_path

from .base import _TestAnomalibDepthDatamodule


class TestMVTec3D(_TestAnomalibDepthDatamodule):
    """MVTec Datamodule Unit Tests."""

    @pytest.fixture
    def datamodule(self, task_type: TaskType) -> MVTec3D:
        # Create and prepare the dataset
        _datamodule = MVTec3D(
            root=get_dataset_path("MVTec3D"),
            category="bagel",
            task=task_type,
            image_size=256,
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
        )
        _datamodule.prepare_data()
        _datamodule.setup()

        return _datamodule
