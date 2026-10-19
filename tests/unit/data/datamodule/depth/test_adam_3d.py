# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - ADAM3D Datamodule."""

from collections.abc import Callable
from pathlib import Path

import pytest
from torchvision.transforms.v2 import Resize

from anomalib.data.datamodules.depth import ADAM3D
from tests.unit.data.datamodule.base.depth import _TestAnomalibDepthDatamodule


class TestADAM3D(_TestAnomalibDepthDatamodule):
    """3D ADAM Datamodule Unit Tests."""

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path, make_dummy_dataset: Callable[[str], Path]) -> ADAM3D:
        """Create and return a 3D ADAM datamodule."""
        make_dummy_dataset("adam_3d")
        datamodule_ = ADAM3D(
            root=dataset_path / "adam_3d",
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
        """Return the path to the data config file."""
        return "examples/configs/data/adam_3d.yaml"

    @staticmethod
    def test_train_samples_mask_path_is_empty_string(datamodule: ADAM3D) -> None:
        """Test if train samples mask path is an empty string and not the string 'nan'."""
        train_samples = datamodule.train_data.samples
        assert (train_samples["mask_path"] == "").all()
        assert not (train_samples["mask_path"] == "nan").any()
