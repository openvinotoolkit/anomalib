# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - LIBAD Datamodule."""

from collections.abc import Callable
from pathlib import Path

import pytest
from pytest_mock import MockerFixture
from torchvision.transforms.v2 import Resize

from anomalib.data import LIBAD
from tests.unit.data.datamodule.base.image import _TestAnomalibImageDatamodule


class TestLIBAD(_TestAnomalibImageDatamodule):
    """LIBAD Datamodule Unit Tests."""

    @staticmethod
    @pytest.mark.parametrize("category", ["1_wrinklng", "../outside"])
    def test_invalid_category_before_download(tmp_path: Path, mocker: MockerFixture, category: str) -> None:
        """Reject invalid categories before checking authentication or downloading."""
        download = mocker.patch("anomalib.data.datamodules.image.libad.hf_hub_download")
        token = mocker.patch("anomalib.data.datamodules.image.libad.get_token")
        datamodule = LIBAD(root=tmp_path, category=category)

        with pytest.raises(ValueError, match="Unknown category"):
            datamodule.prepare_data()

        download.assert_not_called()
        token.assert_not_called()

    @staticmethod
    def test_download_pinned_revision(tmp_path: Path, mocker: MockerFixture) -> None:
        """Download a fixed revision and extract it without network access."""
        mocker.patch("anomalib.data.datamodules.image.libad.module_available", return_value=True)
        mocker.patch("anomalib.data.datamodules.image.libad.get_token", return_value="test-token")
        download = mocker.patch(
            "anomalib.data.datamodules.image.libad.hf_hub_download",
            return_value=str(tmp_path / "LIBAD.zip"),
        )
        extract = mocker.patch("anomalib.data.datamodules.image.libad.extract")

        LIBAD(root=tmp_path).prepare_data()

        assert download.call_args.kwargs["revision"] == "1f4c5f4dbef3b6bb3c122a2340dff5b19014bf74"
        extract.assert_called_once()

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path, make_dummy_dataset: Callable[[str], Path]) -> LIBAD:
        """Create and return a LIBAD datamodule."""
        make_dummy_dataset("libad")
        datamodule_ = LIBAD(
            root=dataset_path / "libad",
            category="1_wrinkling",
            modality="A",
            train_batch_size=4,
            eval_batch_size=4,
            augmentations=Resize((256, 256)),
            test_split_ratio=0.2,
        )

        datamodule_.prepare_data()
        datamodule_.setup()

        return datamodule_

    @pytest.fixture()
    @staticmethod
    def fxt_data_config_path() -> str:
        """Return the path to the test data config."""
        return "examples/configs/data/libad.yaml"
