"""Unit Tests - RealIAD Datamodule."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from torchvision.transforms.v2 import Resize

from anomalib.data import RealIAD
from tests.unit.data.datamodule.base.image import _TestAnomalibImageDatamodule


class TestRealIAD(_TestAnomalibImageDatamodule):
    """RealIAD Datamodule Unit Tests."""

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path) -> RealIAD:
        """Create and return a RealIAD datamodule."""
        _datamodule = RealIAD(
            root=dataset_path / "realiad",
            category="audiojack",
            resolution=256,
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            augmentations=Resize((256, 256)),
        )
        _datamodule.prepare_data()
        _datamodule.setup()

        return _datamodule

    @pytest.fixture()
    @staticmethod
    def fxt_data_config_path() -> str:
        """Return the path to the test data config."""
        return "examples/configs/data/realiad.yaml"
