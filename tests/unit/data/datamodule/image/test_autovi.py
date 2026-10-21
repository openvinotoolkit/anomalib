# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - AutoVI Datamodule."""

from collections.abc import Callable
from pathlib import Path

import pytest
from torchvision.transforms.v2 import Resize

from anomalib.data import AutoVI
from anomalib.data.datasets.image.autovi import make_autovi_dataset
from anomalib.data.utils import Split
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

    @staticmethod
    def test_autovi_split_enum_filtering(dataset_path: Path, make_dummy_dataset: Callable[[str], Path]) -> None:
        """Test if passing a Split enum to make_autovi_dataset works properly."""
        make_dummy_dataset("autovi")
        samples = make_autovi_dataset(dataset_path / "autovi" / "engine_wiring", split=Split.TRAIN)
        assert len(samples) > 0
        assert (samples["split"] == "train").all()
