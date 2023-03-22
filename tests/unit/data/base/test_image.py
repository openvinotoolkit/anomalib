"""Unit Tests - Base Image Datamodules."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from anomalib.data import AnomalibDataModule
from .test_datamodule import _TestAnomalibDataModule


class _TestAnomalibImageDatamodule(_TestAnomalibDataModule):
    @pytest.mark.parametrize("subset", ["train", "val", "test"])
    def test_get_item_returns_correct_keys_and_shapes(self, datamodule: AnomalibDataModule, subset: str) -> None:
        """Test that the datamodule __getitem__ returns image, mask, label and boxes."""

        # Get the dataloader.
        dataloader = getattr(datamodule, f"{subset}_dataloader")()

        # Get the first batch.
        batch = next(iter(dataloader))

        # Check that the batch has the correct shape.
        assert batch["image"].shape == (4, 3, 256, 256)
        assert batch["label"].shape == (4,)

        # TODO: Detection task should return bounding boxes.
        # if dataloader.dataset.task == "detection":
        #     assert batch["boxes"].shape == (4, 4)

        if dataloader.dataset.task in ("detection", "segmentation"):
            assert batch["mask"].shape == (4, 256, 256)
