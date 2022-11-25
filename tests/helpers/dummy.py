import shutil
import tempfile
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
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
    """Creates a very basic CNN model to fit image data for classification task
    The test uses this to check if this model is converted to OpenVINO IR."""

    def __init__(
        self,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 5)
        self.conv3 = nn.Conv2d(32, 1, 7)
        self.fc1 = nn.Linear(400, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = F.dropout(x, p=0.2)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


class DummyLogger(AnomalibTensorBoardLogger):
    def __init__(self):
        self.tempdir = Path(tempfile.mkdtemp())
        super().__init__(name="tensorboard_logs", save_dir=self.tempdir)

    def __del__(self):
        if self.tempdir.exists():
            shutil.rmtree(self.tempdir)
