# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - MPDD Datamodule."""

from collections.abc import Callable
from pathlib import Path

import pytest
from torchvision.transforms.v2 import Resize

from anomalib.data import MPDD
from tests.unit.data.datamodule.base.image import _TestAnomalibImageDatamodule


class TestMPDD(_TestAnomalibImageDatamodule):
    """MPDD Datamodule Unit Tests."""

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path, make_dummy_dataset: Callable[[str], Path]) -> MPDD:
        """Create and return a MPDD datamodule."""
        make_dummy_dataset("mpdd")
        datamodule_ = MPDD(
            root=dataset_path / "mpdd",
            category="dummy",
            train_batch_size=4,
            eval_batch_size=4,
            augmentations=Resize((256, 256)),
        )
        datamodule_.prepare_data()
        datamodule_.setup()

        return datamodule_

    @pytest.fixture()
    @staticmethod
    def fxt_data_config_path() -> str:
        """Return the path to the test data config."""
        return "examples/configs/data/mpdd.yaml"
