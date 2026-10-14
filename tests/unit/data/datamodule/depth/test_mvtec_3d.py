# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - MVTec3D Datamodule."""

from collections.abc import Callable
from pathlib import Path

import pytest
from torchvision.transforms.v2 import Resize

from anomalib.data import MVTec3D
from tests.unit.data.datamodule.base.depth import _TestAnomalibDepthDatamodule


class TestMVTec3D(_TestAnomalibDepthDatamodule):
    """MVTec Datamodule Unit Tests."""

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path, make_dummy_dataset: Callable[[str], Path]) -> MVTec3D:
        """Create and return a Folder 3D datamodule."""
        make_dummy_dataset("mvtec_3d")
        datamodule_ = MVTec3D(
            root=dataset_path / "mvtec_3d",
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
        return "examples/configs/data/mvtec_3d.yaml"
