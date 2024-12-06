"""Unit Tests - Base Image Datamodules."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from anomalib.data import AnomalibDataModule

from .base import _TestAnomalibDataModule


class _TestAnomalibImageDatamodule(_TestAnomalibDataModule):
    # 1. Test if the image datasets are correctly created.

    @staticmethod
    @pytest.mark.parametrize("subset", ["train", "val", "test"])
    def test_get_item_returns_correct_keys_and_shapes(subset: str, datamodule: AnomalibDataModule) -> None:
        """Test that the datamodule __getitem__ returns image, mask, label and boxes."""
        # Get the dataloader.
        dataloader = getattr(datamodule, f"{subset}_dataloader")()

        # Get the first batch.
        batch = next(iter(dataloader))

        # Check that the batch has the correct shape.
        assert batch["image"].shape == (4, 3, 256, 256)
        assert batch["label"].shape == (4,)

        if dataloader.dataset.task in {"detection", "segmentation"}:
            assert batch["mask"].shape == (4, 256, 256)

    @staticmethod
    def test_non_overlapping_splits(datamodule: AnomalibDataModule) -> None:
        """This test ensures that all splits are non-overlapping when split mode == from_test."""
        if datamodule.val_split_mode == "from_test":
            assert (
                len(
                    set(datamodule.test_data.samples["image_path"].values).intersection(
                        set(datamodule.train_data.samples["image_path"].values),
                    ),
                )
                == 0
            ), "Found train and test split contamination"
            assert (
                len(
                    set(datamodule.val_data.samples["image_path"].values).intersection(
                        set(datamodule.test_data.samples["image_path"].values),
                    ),
                )
                == 0
            ), "Found train and test split contamination"

    @staticmethod
    def test_equal_splits(datamodule: AnomalibDataModule) -> None:
        """This test ensures that val and test split are equal when split mode == same_as_test."""
        if datamodule.val_split_mode == "same_as_test":
            assert np.array_equal(
                datamodule.val_data.samples["image_path"].to_numpy(),
                datamodule.test_data.samples["image_path"].to_numpy(),
            ), "Validation and test splits are not equal"
