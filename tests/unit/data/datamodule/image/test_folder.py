"""Unit Tests - Folder Datamodule."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from torchvision.transforms.v2 import Resize

from anomalib.data import Folder
from tests.unit.data.datamodule.base.image import _TestAnomalibImageDatamodule


class TestFolder(_TestAnomalibImageDatamodule):
    """Folder Datamodule Unit Tests.

    All of the folder datamodule tests are placed in ``TestFolder`` class.
    """

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path) -> Folder:
        """Create and return a Folder datamodule."""
        # expects a relative directory to the root.
        mask_dir = "ground_truth/bad"

        # Create and prepare the dataset
        _datamodule = Folder(
            name="dummy",
            root=dataset_path / "mvtecad" / "dummy",
            normal_dir="train/good",
            abnormal_dir="test/bad",
            normal_test_dir="test/good",
            mask_dir=mask_dir,
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            augmentations=Resize((256, 256)),
        )
        _datamodule.setup()

        return _datamodule

    @pytest.fixture()
    @staticmethod
    def fxt_data_config_path() -> str:
        """Return the path to the test data config."""
        return "examples/configs/data/folder.yaml"
