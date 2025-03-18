"""Unit Tests - MVTecLoco Datamodule."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from torchvision.transforms.v2 import Resize

from anomalib.data import MVTecLOCO
from tests.unit.data.datamodule.base.image import _TestAnomalibImageDatamodule


class TestMVTecLOCO(_TestAnomalibImageDatamodule):
    """MVTecLOCO Datamodule Unit Tests."""

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path) -> MVTecLOCO:
        """Create and return a MVTecLOCO datamodule."""
        datamodule = MVTecLOCO(
            root=dataset_path / "mvtec_loco",
            category="dummy",
            train_batch_size=4,
            eval_batch_size=4,
            augmentations=Resize((256, 256)),
        )
        datamodule.prepare_data()
        datamodule.setup()

        return datamodule

    @pytest.fixture()
    @staticmethod
    def fxt_data_config_path() -> str:
        """Return the path to the test data config."""
        return "examples/configs/data/mvtec_loco.yaml"

    @staticmethod
    def test_mask_is_binary(datamodule: MVTecLOCO) -> None:
        """Test if the mask tensor is binary."""
        mask_tensor = datamodule.test_data[0].gt_mask
        if mask_tensor is not None:
            is_binary = (mask_tensor.eq(0) | mask_tensor.eq(1)).all()
            assert is_binary.item() is True
