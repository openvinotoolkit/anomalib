import shutil
import tempfile
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from anomalib.utils.loggers.tensorboard import AnomalibTensorBoardLogger


class DummyDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.ones(1)


class DummyDataModule(pl.LightningDataModule):
    def train_dataloader(self) -> DataLoader:
        return DataLoader(DummyDataset())

    def val_dataloader(self) -> DataLoader:
        return DataLoader(DummyDataset())

    def test_dataloader(self) -> DataLoader:
        return DataLoader(DummyDataset())


class DummyModel(nn.Module):
    pass


class DummyLogger(AnomalibTensorBoardLogger):
    def __init__(self):
        self.tempdir = Path(tempfile.mkdtemp())
        super().__init__(name="tensorboard_logs", save_dir=self.tempdir)

    def __del__(self):
        if self.tempdir.exists():
            shutil.rmtree(self.tempdir)
