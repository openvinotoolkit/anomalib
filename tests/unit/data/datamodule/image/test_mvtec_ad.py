# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - MVTecAD Datamodule."""

from pathlib import Path

import pytest
from torchvision.transforms.v2 import Resize

from anomalib.data import MVTecAD
from tests.unit.data.datamodule.base.image import _TestAnomalibImageDatamodule


class TestMVTecAD(_TestAnomalibImageDatamodule):
    """MVTec Datamodule Unit Tests."""

    @pytest.fixture()
    @staticmethod
    def datamodule(mvtecad_path: Path) -> MVTecAD:
        """Create and return a MVTec datamodule."""
        datamodule_ = MVTecAD(
            root=mvtecad_path,
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
        return "examples/configs/data/mvtec.yaml"
