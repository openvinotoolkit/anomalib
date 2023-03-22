"""Unit Tests - Base Video Datamodules."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import random

from anomalib.data import AnomalibDataModule

from .test_datamodule import _TestAnomalibDataModule


class _TestAnomalibVideoDatamodule(_TestAnomalibDataModule):
    def test_get_item_returns_correct_keys_and_shapes(self, datamodule: AnomalibDataModule) -> None:
        """Test that the datamodule __getitem__ returns image, mask, label and boxes."""

        # Randomly select a subset of the dataset.
        subsets = ["train", "val", "test"]
        random_subset = random.choice(subsets)

        # Get the dataloader.
        dataloader = getattr(datamodule, f"{random_subset}_dataloader")()

        # Get the first batch.
        batch = next(iter(dataloader))

        # Check that the batch has the correct keys.
        expected_train_keys = {"image", "video_path", "frames", "last_frame", "original_image"}
        expected_eval_keys = expected_train_keys | {"label", "mask"}
        expected_eval_detection_keys = expected_eval_keys | {"boxes"}

        if random_subset == "train":
            expected_keys = expected_train_keys
        else:
            if dataloader.dataset.task == "detection":
                expected_keys = expected_eval_detection_keys
            else:
                expected_keys = expected_eval_keys

        assert batch.keys() == expected_keys

        # Check that the batch has the correct shape.
        assert batch["image"].shape == (4, 3, 256, 256)
        assert len(batch["video_path"]) == 4
        assert len(batch["frames"]) == 4
        assert len(batch["last_frame"]) == 4
        # We don't know the shape of the original image, so we only check that it is a list of 4 images.
        assert batch["original_image"].shape[0] == 4

        if random_subset in ("val", "test"):
            assert len(batch["label"]) == 4
            assert batch["mask"].shape == (4, 256, 256)
