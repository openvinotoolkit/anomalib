from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch import optim
from torch.utils.data import DataLoader, Dataset

from anomalib.core.callbacks.visualizer_callback import VisualizerCallback


class XORDataset(Dataset):
    def __init__(self):
        super(XORDataset, self).__init__()
        self.x = torch.tensor([[0, 0], [1, 1], [1, 0], [0, 1]], dtype=torch.float32)
        self.y = torch.tensor([0, 0, 1, 1], dtype=torch.float32)

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y


class XORDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 2):
        super(XORDataModule, self).__init__()
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(XORDataset(), batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(XORDataset(), batch_size=self.batch_size)


class DummyModel(pl.LightningModule):
    """A dummy model which fits xor problem"""

    def __init__(self, hparams: Union[DictConfig, ListConfig]):
        super(DummyModel, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)
        self.loss_fn = nn.MSELoss()
        self.callbacks = [VisualizerCallback()]  # test if this is removed
        self.save_hyperparameters(hparams)

    def forward(self, x):
        x = self.fc1(x)
        x = F.dropout(x, p=self.hparams.model.dropout)
        x = self.fc2(x)
        return x

    def training_step(self, batch, _):
        x, y = batch
        y_hat = F.sigmoid(self.forward(x))
        loss = self.loss_fn(y_hat, y)
        return {"loss": loss}

    def test_step(self, batch, _):
        x, y = batch
        y_hat = F.sigmoid(self.forward(x))
        loss = self.loss_fn(y_hat, y)
        self.log(name="loss", value=loss.item(), prog_bar=True)

    def configure_optimizers(self):
        return optim.SGD(
            self.parameters(),
            lr=self.hparams.model.lr,
            momentum=self.hparams.model.momentum,
            weight_decay=self.hparams.model.weight_decay,
        )
