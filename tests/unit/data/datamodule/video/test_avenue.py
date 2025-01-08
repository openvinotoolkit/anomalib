"""Unit Tests - Avenue Datamodule."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from torchvision.transforms.v2 import Resize

from anomalib.data import Avenue
from tests.unit.data.datamodule.base.video import _TestAnomalibVideoDatamodule


class TestAvenue(_TestAnomalibVideoDatamodule):
    """Avenue Datamodule Unit Tests."""

    @pytest.fixture()
    @staticmethod
    def clip_length_in_frames() -> int:
        """Return the number of frames in each clip."""
        return 2

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path, clip_length_in_frames: int) -> Avenue:
        """Create and return a Avenue datamodule."""
        _datamodule = Avenue(
            root=dataset_path / "avenue",
            gt_dir=dataset_path / "avenue" / "ground_truth_demo",
            clip_length_in_frames=clip_length_in_frames,
            num_workers=0,
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
        return "examples/configs/data/avenue.yaml"
