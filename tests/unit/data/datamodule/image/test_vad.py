"""Unit Tests - VAD Datamodule."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from torchvision.transforms.v2 import Resize

from anomalib.data import VAD
from tests.unit.data.datamodule.base.image import _TestAnomalibImageDatamodule


class TestVAD(_TestAnomalibImageDatamodule):
    """VAD Datamodule Unit Tests."""

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path) -> VAD:
        """Create and return a VAD datamodule."""
        _datamodule = VAD(
            root=dataset_path / "vad",
            category="vad",
            train_batch_size=4,
            eval_batch_size=4,
            augmentations=Resize((256, 256)),
        )
        _datamodule.prepare_data()
        _datamodule.setup()

        return _datamodule

    @pytest.fixture()
    @staticmethod
    def fxt_data_config_path() -> str:
        """Return the path to the test data config."""
        return "examples/configs/data/vad.yaml"
