"""Unit Tests - Folder3D Datamodule."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from anomalib.data import Folder3D, TaskType
import pytest
from .base import _TestAnomalibDepthDatamodule


class TestFolder3D(_TestAnomalibDepthDatamodule):
    """Folder3D Datamodule Unit Tests."""

    @pytest.fixture
    def datamodule(self, task_type: TaskType) -> Folder3D:
        # Create and prepare the dataset
        _datamodule = Folder3D(
            normal_dir="./train/good/rgb",
            root="./datasets/MVTec3D/bagel",
            abnormal_dir="./test/combined/rgb",
            normal_test_dir="./test/good/rgb",
            mask_dir="./test/combined/gt",
            normal_depth_dir="./train/good/xyz",
            abnormal_depth_dir="./test/combined/xyz",
            normal_test_depth_dir="./test/good/xyz",
            image_size=256,
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
        )
        _datamodule.prepare_data()
        _datamodule.setup()

        return _datamodule
