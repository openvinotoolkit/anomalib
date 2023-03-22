"""Unit Tests - Folder Datamodule."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from anomalib.data import Folder, TaskType
from tests.helpers.dataset import get_dataset_path

from .base import _TestAnomalibImageDatamodule


class TestFolder(_TestAnomalibImageDatamodule):
    """Folder Datamodule Unit Tests.

    All of the folder datamodule tests are placed in ``TestFolder`` class.
    """

    @pytest.fixture(params=[None, "good_test"])
    def normal_test_dir(self, request) -> str:
        """Create and return a normal test directory."""
        return request.param

    @pytest.fixture
    def datamodule(self, task_type: TaskType, normal_test_dir: str) -> Folder:
        """Create and return a folder datamodule.

        This datamodule uses the MVTec bottle dataset for testing.
        """
        # Make sure to use a mask directory for segmentation. Folder datamodule
        # expects a relative directory to the root.
        mask_dir = None if task_type == TaskType.CLASSIFICATION else "ground_truth/broken_large"

        # Create and prepare the dataset
        _datamodule = Folder(
            root=get_dataset_path("bottle"),
            normal_dir="good",
            abnormal_dir="broken_large",
            normal_test_dir=normal_test_dir,
            mask_dir=mask_dir,
            image_size=(256, 256),
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            task=task_type,
        )
        _datamodule.setup()

        return _datamodule
