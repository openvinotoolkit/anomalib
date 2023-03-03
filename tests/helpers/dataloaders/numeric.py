import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class DummyNumericDataset(Dataset):
    """Returns a single tensor of ones."""

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.ones(1)


class DummyNumericDataModule(pl.LightningDataModule):
    """Returns a single tensor of ones."""

    def train_dataloader(self) -> DataLoader:
        return DataLoader(DummyNumericDataset())

    def val_dataloader(self) -> DataLoader:
        return DataLoader(DummyNumericDataset())

    def test_dataloader(self) -> DataLoader:
        return DataLoader(DummyNumericDataset())
