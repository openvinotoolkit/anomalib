"""Anomalib datamodule base class."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from abc import ABC
from typing import Any, Dict, List, Optional

from pandas import DataFrame
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, default_collate

from anomalib.data.base.dataset import AnomalibDataset
from anomalib.data.utils import ValSplitMode, random_split

logger = logging.getLogger(__name__)


def collate_fn(batch: List) -> Dict[str, Any]:
    """Custom collate function that collates bounding boxes as lists.

    Bounding boxes are collated as a list of tensors, while the default collate function is used for all other entries.

    Args:
        batch (List): list of items in the batch where len(batch) is equal to the batch size.

    Returns:
        Dict[str, Any]: Dictionary containing the collated batch information.
    """
    elem = batch[0]  # sample an element from the batch to check the type.
    out_dict = {}
    if isinstance(elem, dict):
        if "boxes" in elem.keys():
            # collate boxes as list
            out_dict["boxes"] = [item.pop("boxes") for item in batch]
        # collate other data normally
        out_dict.update({key: default_collate([item[key] for item in batch]) for key in elem})
        return out_dict
    return default_collate(batch)


class AnomalibDataModule(LightningDataModule, ABC):
    """Base Anomalib data module.

    Args:
        train_batch_size (int): Batch size used by the train dataloader.
        test_batch_size (int): Batch size used by the val and test dataloaders.
        num_workers (int): Number of workers used by the train, val and test dataloaders.
        seed (Optional[int], optional): Seed used during random subset splitting.
    """

    def __init__(
        self,
        train_batch_size: int,
        eval_batch_size: int,
        num_workers: int,
        val_split_mode: ValSplitMode,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.val_split_mode = val_split_mode
        self.seed = seed

        self.train_data: Optional[AnomalibDataset] = None
        self.val_data: Optional[AnomalibDataset] = None
        self.test_data: Optional[AnomalibDataset] = None

        self._samples: Optional[DataFrame] = None

    def setup(self, stage: Optional[str] = None):
        """Setup train, validation and test data.

        Args:
          stage: Optional[str]:  Train/Val/Test stages. (Default value = None)
        """
        if not self.is_setup:
            self._setup(stage)
        assert self.is_setup

    def _setup(self, _stage: Optional[str] = None) -> None:
        """Set up the datasets and perform dynamic subset splitting.

        This method may be overridden in subclass for custom splitting behaviour.

        Note: The stage argument is not used here. This is because, for a given instance of an AnomalibDataModule
        subclass, all three subsets are created at the first call of setup(). This is to accommodate the subset
        splitting behaviour of anomaly tasks, where the validation set is usually extracted from the test set, and
        the test set must therefore be created as early as the `fit` stage.
        """
        assert self.train_data is not None
        assert self.test_data is not None

        self.train_data.setup()
        self.test_data.setup()
        if self.val_split_mode == ValSplitMode.FROM_TEST:
            self.val_data, self.test_data = random_split(self.test_data, [0.5, 0.5], label_aware=True, seed=self.seed)
        elif self.val_split_mode == ValSplitMode.SAME_AS_TEST:
            self.val_data = self.test_data
        elif self.val_split_mode != ValSplitMode.NONE:
            raise ValueError(f"Unknown validation split mode: {self.val_split_mode}")

    @property
    def is_setup(self):
        """Checks if setup() has been called."""
        # at least one of [train_data, val_data, test_data] should be setup
        if self.train_data is not None and self.train_data.is_setup:
            return True
        if self.val_data is not None and self.val_data.is_setup:
            return True
        if self.test_data is not None and self.test_data.is_setup:
            return True
        return False

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Get train dataloader."""
        return DataLoader(
            dataset=self.train_data, shuffle=True, batch_size=self.train_batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Get validation dataloader."""
        return DataLoader(
            dataset=self.val_data,
            shuffle=False,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Get test dataloader."""
        return DataLoader(
            dataset=self.test_data,
            shuffle=False,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
