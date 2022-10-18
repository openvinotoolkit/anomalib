from typing import Union

import pytorch_lightning as pl
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData

from anomalib.utils.callbacks import ImageVisualizerCallback
from anomalib.utils.metrics import (
    AnomalyScoreDistribution,
    AnomalyScoreThreshold,
    MinMax,
)


class FakeDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super(FakeDataModule, self).__init__()
        self.batch_size = batch_size
        self.pre_process = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def train_dataloader(self):
        return DataLoader(
            FakeData(
                size=1000,
                num_classes=10,
                transform=self.pre_process,
                image_size=(3, 32, 32),
            ),
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            FakeData(
                size=100,
                num_classes=10,
                transform=self.pre_process,
                image_size=(3, 32, 32),
            ),
            batch_size=self.batch_size,
        )


class DummyModel(nn.Module):
    """Creates a very basic CNN model to fit image data for classification task
    The test uses this to check if this model is converted to OpenVINO IR."""

    def __init__(self, hparams: Union[DictConfig, ListConfig]):
        super().__init__()
        self.hparams = hparams
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
        x = F.dropout(x, p=self.hparams.model.dropout)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


class DummyLightningModule(pl.LightningModule):
    """A dummy model which fits the torchvision FakeData dataset."""

    def __init__(self, hparams: Union[DictConfig, ListConfig]):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.loss_fn = nn.NLLLoss()
        self.callbacks = [
            ImageVisualizerCallback(
                mode="full",
                task="segmentation",
                image_save_path=hparams.project.path + "/images",
                log_images=False,
                save_images=True,
            )
        ]  # test if this is removed

        self.image_threshold = AnomalyScoreThreshold(hparams.model.threshold.image_default).cpu()
        self.pixel_threshold = AnomalyScoreThreshold(hparams.model.threshold.pixel_default).cpu()

        self.training_distribution = AnomalyScoreDistribution().cpu()
        self.min_max = MinMax().cpu()
        self.model = DummyModel(hparams)

    def training_step(self, batch, _):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return {"loss": loss}

    def validation_step(self, batch, _):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log(name="loss", value=loss.item(), prog_bar=True)

    def configure_optimizers(self):
        return optim.SGD(
            self.parameters(),
            lr=self.hparams.model.lr,
            momentum=self.hparams.model.momentum,
            weight_decay=self.hparams.model.weight_decay,
        )
