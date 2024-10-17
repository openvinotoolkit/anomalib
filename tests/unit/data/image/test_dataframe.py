"""Unit Tests - Dataframe Datamodule."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from anomalib import TaskType
from anomalib.data import Dataframe, Folder
from tests.unit.data.base.image import _TestAnomalibImageDatamodule


class TestDataframe(_TestAnomalibImageDatamodule):
    """Dataframe Datamodule Unit Tests.

    All of the Dataframe datamodule tests are placed in ``TestDataframe`` class.
    """

    @staticmethod
    def get_samples_dataframe(dataset_path: Path, task_type: TaskType) -> pd.DataFrame:
        """Create samples pd.DataFrame."""
        # Make sure to use a mask directory for segmentation. Dataframe datamodule
        # expects a relative directory to the root.
        mask_dir = None if task_type == TaskType.CLASSIFICATION else "ground_truth/bad"

        # Create folder datamodule to get samples dataframe
        _folder_datamodule = Folder(
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
        _folder_datamodule.setup()
        _samples = pd.concat([
            _folder_datamodule.train_data.samples,
            _folder_datamodule.test_data.samples,
            _folder_datamodule.val_data.samples,
        ])

        # drop label column as it is inferred from the other columns
        return _samples.drop(["label"], axis="columns")

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path, task_type: TaskType) -> Dataframe:
        """Create and return a Dataframe datamodule."""
        # Create and prepare the dataset
        _samples = TestDataframe.get_samples_dataframe(dataset_path, task_type)
        _datamodule = Dataframe(
            name="dummy",
            samples=_samples,
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
        return "configs/data/dataframe.yaml"


class TestDataframeFromFile(TestDataframe):
    """Dataframe Datamodule Unit Tests for alternative constructor.

    Tests for the Datamodule creation from file.
    """

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path, task_type: TaskType) -> Dataframe:
        """Create and return a Dataframe datamodule."""
        # Create and prepare the dataset
        _samples = TestDataframeFromFile.get_samples_dataframe(dataset_path, task_type)
        with tempfile.TemporaryFile() as samples_file:
            _samples.to_csv(samples_file)
            samples_file.seek(0)

            _datamodule = Dataframe.from_file(
                name="dummy",
                file_path=samples_file,
                train_batch_size=4,
                eval_batch_size=4,
                num_workers=0,
                task=task_type,
            )
            _datamodule.setup()

        return _datamodule
