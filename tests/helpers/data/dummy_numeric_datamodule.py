"""Tests - NUmeric Data Helpers."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class DummyNumericDataset(Dataset):
    """Returns a single tensor of ones."""

    def __len__(self):
        return 1

    def __getitem__(self, idx) -> Tensor:
        return torch.ones(1)


class DummyNumericDataModule(pl.LightningDataModule):
    """Returns a single tensor of ones."""

    def train_dataloader(self) -> DataLoader:
        return DataLoader(DummyNumericDataset())

    def val_dataloader(self) -> DataLoader:
        return DataLoader(DummyNumericDataset())

    def test_dataloader(self) -> DataLoader:
        return DataLoader(DummyNumericDataset())
