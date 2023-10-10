from typing import Union
from unittest.mock import MagicMock

import lightning.pytorch as pl
import torch
from omegaconf import DictConfig, ListConfig
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData

from anomalib.utils.callbacks import ImageVisualizerCallback
from anomalib.utils.metrics import AnomalyScoreDistribution, F1AdaptiveThreshold, MinMax
from anomalib.utils.metrics.collection import AnomalibMetricCollection
from tests.helpers.dummy import DummyModel


class FakeDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
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


class DummyLightningModule(pl.LightningModule):
    """A dummy model which fits the torchvision FakeData dataset."""

    def __init__(self, hparams: Union[DictConfig, ListConfig]):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.loss_fn = nn.NLLLoss()

        self.image_threshold = F1AdaptiveThreshold(hparams.model.init_args.threshold.image_default).cpu()
        self.pixel_threshold = F1AdaptiveThreshold(hparams.model.init_args.threshold.pixel_default).cpu()

        self.training_distribution = AnomalyScoreDistribution().cpu()
        self.min_max = MinMax().cpu()
        self.model = DummyModel()
        self.image_metrics = MagicMock()
        self.pixel_metrics = MagicMock()

    def training_step(self, batch, _):
        x, y = batch["image"], batch["label"]
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return {"loss": loss}

    def validation_step(self, batch, _):
        x, y = batch["image"], batch["label"]
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log(name="loss", value=loss.item(), prog_bar=True)
        return {"anomaly_maps": batch["image"], "label": batch["label"], "pred_scores": torch.tensor([1.0])}

    def configure_optimizers(self):
        return optim.SGD(
            self.parameters(),
            lr=self.hparams.model.init_args.lr,
            momentum=self.hparams.model.init_args.momentum,
            weight_decay=self.hparams.model.init_args.weight_decay,
        )

    @property
    def trainer_arguments(self) -> dict:
        return {}
