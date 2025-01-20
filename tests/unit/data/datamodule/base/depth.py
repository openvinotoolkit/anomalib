"""Unit Tests - Base Depth Datamodules."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields

import pytest

from anomalib.data import AnomalibDataModule
from tests.unit.data.datamodule.base.base import _TestAnomalibDataModule


class _TestAnomalibDepthDatamodule(_TestAnomalibDataModule):
    @staticmethod
    @pytest.mark.parametrize("subset", ["train", "val", "test"])
    def test_get_item_returns_correct_keys_and_shapes(subset: str, datamodule: AnomalibDataModule) -> None:
        """Test that the datamodule __getitem__ returns the correct keys and shapes."""
        # Get the dataloader.
        dataloader = getattr(datamodule, f"{subset}_dataloader")()

        # Get the first batch.
        batch = next(iter(dataloader))

        # Check that the batch has the correct keys.
        expected_fields = {"image_path", "depth_path", "gt_label", "image", "depth_map"}

        if dataloader.dataset.task == "segmentation":
            expected_fields.add("gt_mask")
            # Add mask_path to expected fields if it's present in the batch
            if hasattr(batch, "mask_path") and batch.mask_path is not None:
                expected_fields.add("mask_path")

        batch_fields = {field.name for field in fields(batch) if getattr(batch, field.name) is not None}
        assert batch_fields == expected_fields

        # Check that the batch has the correct shape.
        assert len(batch.image_path) == 4
        assert len(batch.depth_path) == 4
        assert batch.image.shape == (4, 3, 256, 256)
        assert batch.depth_map.shape == (4, 3, 256, 256)
        assert batch.gt_label.shape == (4,)

        if dataloader.dataset.task == "segmentation":
            assert batch.gt_mask.shape == (4, 256, 256)
