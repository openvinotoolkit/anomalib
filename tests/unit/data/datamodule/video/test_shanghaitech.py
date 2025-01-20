"""Unit Tests - ShanghaiTech Datamodule."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from torchvision.transforms.v2 import Resize

from anomalib.data import ShanghaiTech
from tests.unit.data.datamodule.base.video import _TestAnomalibVideoDatamodule


class TestShanghaiTech(_TestAnomalibVideoDatamodule):
    """ShanghaiTech Datamodule Unit Tests."""

    @pytest.fixture()
    @staticmethod
    def clip_length_in_frames() -> int:
        """Return the number of frames in each clip."""
        return 2

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path, clip_length_in_frames: int) -> ShanghaiTech:
        """Create and return a Shanghai datamodule."""
        _datamodule = ShanghaiTech(
            root=dataset_path / "shanghaitech",
            scene=1,
            clip_length_in_frames=clip_length_in_frames,
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
        return "examples/configs/data/shanghaitech.yaml"
