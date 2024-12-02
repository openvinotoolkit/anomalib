"""Base Video Data Module."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

from anomalib.data.dataclasses import VideoBatch
from anomalib.data.utils import ValSplitMode

from .image import AnomalibDataModule


class AnomalibVideoDataModule(AnomalibDataModule):
    """Base class for video data modules."""

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform dynamic subset splitting.

        This method may be overridden in subclass for custom splitting behaviour.

        Video datamodules are not compatible with synthetic anomaly generation.
        """
        if self.train_data is None:
            msg = "self.train_data cannot be None."
            raise ValueError(msg)

        if self.test_data is None:
            msg = "self.test_data cannot be None."
            raise ValueError(msg)

        self.train_data.setup()
        self.test_data.setup()

        if self.val_split_mode == ValSplitMode.SYNTHETIC:
            msg = f"Val split mode {self.test_split_mode} not supported for video datasets."
            raise ValueError(msg)

    def _post_setup(self) -> None:
        """Processing video datasets has not been implemented yet."""

    @property
    def collate_fn(self) -> Callable:
        """Return the collate function for video batches."""
        return VideoBatch.collate
