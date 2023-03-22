"""Tensor Dummy Data Helpers.

Dummy dataset that returns a single anomalous image of size 32x32 with a
10x10 white square in the middle.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class DummyTensorDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.image = torch.zeros((3, 32, 32), dtype=torch.float32)
        self.image[:, 5:15, 5:15] = 1.0

    def __getitem__(self, index):
        return {"images": self.image, "label": torch.tensor(0)}

    def __len__(self):
        return 2


class DummyTensorDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.dataset = DummyTensorDataset()

    def train_dataloader(self):
        return DataLoader(self.dataset, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.dataset, shuffle=False, num_workers=36)

    def test_dataloader(self):
        return DataLoader(self.dataset, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.dataset, shuffle=False)
