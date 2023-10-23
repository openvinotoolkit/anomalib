"""Unit Tests - Datamodules."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from anomalib.data import MVTec


def make_mvtec_data_module(
    root: Path,
    task: str = "segmentation",
    batch_size: int = 1,
    test_split_mode: str = "from_dir",
    val_split_mode: str = "same_as_test",
) -> MVTec:
    """Create a dummy MVTec AD Data Module."""
    data_module = MVTec(
        root=root,
        category="shapes",
        image_size=(256, 256),
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=0,
        task=task,
        test_split_mode=test_split_mode,
        val_split_mode=val_split_mode,
    )
    data_module.setup()
    return data_module


class TestDataModule:
    """Test MVTec AD Data Module."""

    @pytest.mark.parametrize("batch_size", [1, 8])
    def test_batch_size(self, dataset_path: Path, batch_size: int) -> None:
        """Test if both single and multiple batch size returns outputs with the right shape."""
        # TODO (samet-akcay): Convert ``make_mvtec_data_module`` to a fixture.
        data_module = make_mvtec_data_module(root=dataset_path, batch_size=batch_size)
        _, train_data_sample = next(enumerate(data_module.train_dataloader()))
        _, val_data_sample = next(enumerate(data_module.val_dataloader()))
        assert train_data_sample["image"].shape[0] == batch_size
        assert val_data_sample["image"].shape[0] == batch_size
