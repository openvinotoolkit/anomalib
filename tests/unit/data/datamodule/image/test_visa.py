# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - Visa Datamodule."""

from collections.abc import Callable
from pathlib import Path

import pytest
from torchvision.transforms.v2 import Resize

from anomalib.data import Visa
from tests.unit.data.datamodule.base.image import _TestAnomalibImageDatamodule


class TestVisa(_TestAnomalibImageDatamodule):
    """Visa Datamodule Unit Tests."""

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path, make_dummy_dataset: Callable[[str], Path]) -> Visa:
        """Create and return a Avenue datamodule."""
        make_dummy_dataset("visa")
        datamodule_ = Visa(
            root=dataset_path,
            category="dummy",
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            augmentations=Resize((256, 256)),
        )
        datamodule_.prepare_data()
        datamodule_.setup()

        return datamodule_

    @pytest.fixture()
    @staticmethod
    def fxt_data_config_path() -> str:
        """Return the path to the test data config."""
        return "examples/configs/data/visa.yaml"
