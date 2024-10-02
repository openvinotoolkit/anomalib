"""Unit Tests - Folder Datamodule."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from anomalib import TaskType
from anomalib.data import Folder
from tests.unit.data.base.image import _TestAnomalibImageDatamodule


class TestFolder(_TestAnomalibImageDatamodule):
    """Folder Datamodule Unit Tests.

    All of the folder datamodule tests are placed in ``TestFolder`` class.
    """

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path, task_type: TaskType) -> Folder:
        """Create and return a Folder datamodule."""
        # Make sure to use a mask directory for segmentation. Folder datamodule
        # expects a relative directory to the root.
        mask_dir = None if task_type == TaskType.CLASSIFICATION else "ground_truth/bad"

        # Create and prepare the dataset
        _datamodule = Folder(
            name="dummy",
            root=dataset_path / "mvtec" / "dummy",
            normal_dir="train/good",
            abnormal_dir="test/bad",
            normal_test_dir="test/good",
            mask_dir=mask_dir,
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            task=task_type,
        )
        _datamodule.setup()

        return _datamodule

    @pytest.fixture()
    @staticmethod
    def fxt_data_config_path() -> str:
        """Return the path to the test data config."""
        return "configs/data/folder.yaml"
