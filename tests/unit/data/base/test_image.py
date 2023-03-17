"""Unit Tests - Base Image Datamodules."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from anomalib.data import AnomalibDataModule
import random
from .test_datamodule import _TestAnomalibDataModule


class _TestAnomalibImageDatamodule(_TestAnomalibDataModule):
    def test_get_item_returns_correct_shapes(self, datamodule: AnomalibDataModule) -> None:
        """Test that the datamodule __getitem__ returns image, mask, label and boxes."""

        # Randomly select a subset of the dataset.
        subsets = ["train", "val", "test"]
        random_subset = random.choice(subsets)

        # Get the dataloader.
        dataloader = getattr(datamodule, f"{random_subset}_dataloader")()

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
