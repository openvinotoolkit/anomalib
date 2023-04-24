"""Dummy dataset that returns a single anomalous image of size 32x32 with a 10x10 white square in the middle."""
import copy
from typing import List

import torch
from einops import reduce
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class DummyTensorDataset(Dataset):
    def __init__(self):
        super().__init__()
        normal_image = torch.zeros((3, 32, 32), dtype=torch.float32)
        abnormal_image = torch.zeros((3, 32, 32), dtype=torch.float32)
        abnormal_image[:, 5:15, 5:15] = 1.0
        self.images = [normal_image, abnormal_image]
        self.labels = torch.tensor([0, 1])
        self.mask = reduce(abnormal_image.type(torch.uint8), "c h w -> 1 h w", "max")

    def __getitem__(self, index):
        item = {"image": self.images[index], "label": self.labels[index]}
        if self.labels[index] == 1:
            item["mask"] = self.mask
        return item

    def __len__(self):
        return len(self.images)

    def get_subset(self, data: str) -> "DummyTensorDataset":
        """Get's a subset of the dataset.

        Args:
            data (str): Get's a subset of the dataset. Must be "normal" or "abnormal".

        Returns:
            DummyTensorDataset: Subset of the dataset.
        """
        if data == "normal":
            index = 0
        elif data == "abnormal":
            index = 1
        else:
            raise ValueError(f"Invalid data type {data}. Must be 'normal' or 'abnormal'.")

        dataset = copy.deepcopy(self)
        dataset.images = [dataset.images[index]]
        dataset.labels = [dataset.labels[index]]
        return dataset


class DummyTensorDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.dataset = DummyTensorDataset()

    def train_dataloader(self):
        return DataLoader(self.dataset.get_subset("normal"), shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.dataset, shuffle=False, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.dataset, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.dataset.get_subset("abnormal"), shuffle=False)
