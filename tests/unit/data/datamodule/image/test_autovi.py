# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - AutoVI Datamodule."""

from collections.abc import Callable
from pathlib import Path

import pytest
from torchvision.transforms.v2 import Resize

from anomalib.data import AutoVI
from tests.unit.data.datamodule.base.image import _TestAnomalibImageDatamodule


class TestAutoVI(_TestAnomalibImageDatamodule):
    """AutoVI Datamodule Unit Tests."""

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path, make_dummy_dataset: Callable[[str], Path]) -> AutoVI:
        """Create and return an AutoVI datamodule."""
        make_dummy_dataset("autovi")
        datamodule_ = AutoVI(
            root=dataset_path / "autovi",
            category="engine_wiring",
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
        return "examples/configs/data/autovi.yaml"
