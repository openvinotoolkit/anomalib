"""Unit Tests - Folder3D Datamodule."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from anomalib import TaskType
from anomalib.data import Folder3D
from tests.unit.data.base import _TestAnomalibDepthDatamodule


class TestFolder3D(_TestAnomalibDepthDatamodule):
    """Folder3D Datamodule Unit Tests."""

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path, task_type: TaskType) -> Folder3D:
        """Create and return a Folder 3D datamodule."""
        _datamodule = Folder3D(
            name="dummy",
            root=dataset_path / "mvtec_3d/dummy",
            normal_dir="train/good/rgb",
            abnormal_dir="test/bad/rgb",
            normal_test_dir="test/good/rgb",
            mask_dir="test/bad/gt",
            normal_depth_dir="train/good/xyz",
            abnormal_depth_dir="test/bad/xyz",
            normal_test_depth_dir="test/good/xyz",
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
        return "configs/data/folder_3d.yaml"

    @staticmethod
    def test_datamodule_from_config(fxt_data_config_path: str) -> None:
        """Test method to create a datamodule from a configuration file.

        Args:
            fxt_data_config_path (str): The path to the configuration file.

        Returns:
            None
        """
        pytest.skip("The configuration file does not exist.")
        _ = fxt_data_config_path
