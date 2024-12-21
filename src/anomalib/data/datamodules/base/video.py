"""Base Video Data Module.

This module provides the base data module class for video datasets in Anomalib.
It extends :class:`AnomalibDataModule` with video-specific functionality.

The module contains:
    - :class:`AnomalibVideoDataModule`: Base class for all video data modules

Example:
    Create a video datamodule from a config file::

        >>> from anomalib.data import AnomalibVideoDataModule
        >>> data_config = "examples/configs/data/ucsd_ped.yaml"
        >>> datamodule = AnomalibVideoDataModule.from_config(config_path=data_config)
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from anomalib.data.utils import ValSplitMode

from .image import AnomalibDataModule


class AnomalibVideoDataModule(AnomalibDataModule):
    """Base class for video data modules.

    This class extends :class:`AnomalibDataModule` to handle video datasets.
    Unlike image datasets, video datasets do not support dynamic test split
    assignment or synthetic anomaly generation.
    """

    def _create_test_split(self) -> None:
        """Video datamodules do not support dynamic assignment of test split.

        Video datasets typically come with predefined train/test splits due to
        temporal dependencies between frames.
        """

    def _setup(self, _stage: str | None = None) -> None:
        """Set up video datasets and perform validation split.

        This method initializes the train and test datasets and creates the
        validation split if specified. It ensures that both train and test
        datasets are properly defined and configured.

        Args:
            _stage: Current stage of training. Defaults to ``None``.

        Raises:
            ValueError: If ``train_data`` or ``test_data`` is ``None``.
            ValueError: If ``val_split_mode`` is set to ``SYNTHETIC``.
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

        self._create_val_split()
