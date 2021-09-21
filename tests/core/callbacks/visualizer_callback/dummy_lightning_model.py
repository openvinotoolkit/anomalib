from pathlib import Path
from typing import Tuple, Union
from unittest import mock

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch.utils.data import DataLoader, Dataset

from anomalib.core.callbacks.visualizer_callback import VisualizerCallback
from anomalib.core.utils.anomaly_map_generator import BaseAnomalyMapGenerator
from anomalib.models.base.lightning_modules import BaseAnomalyLightning
from anomalib.models.base.torch_modules import BaseAnomalyModule


class DummyDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.ones(1)


class DummyDataModule(pl.LightningDataModule):
    def test_dataloader(self) -> DataLoader:
        return DataLoader(DummyDataset())


class DummyAnomalyMapGenerator(BaseAnomalyMapGenerator):
    def __init__(self):
        super().__init__(input_size=(100, 100), alpha=0.4, gamma=0, sigma=4)


class DummyModel(BaseAnomalyModule):
    def __init__(self):
        super().__init__()
        self.anomaly_map_generator = DummyAnomalyMapGenerator()


class DummyModule(BaseAnomalyLightning):
    """A dummy model which calls visualizer callback on fake images and masks"""

    def __init__(self, hparams: Union[DictConfig, ListConfig]):
        super().__init__(hparams)
        self.model = DummyModel()
        self.task = "segmentation"
        self.callbacks = [VisualizerCallback()]  # test if this is removed
        self.filenames = [Path("test1.jpg"), Path("test2.jpg")]
        self.images = [torch.rand((1, 3, 100, 100))] * 2
        self.true_masks = np.zeros((2, 100, 100))
        self.anomaly_maps = np.ones((2, 100, 100))

    def test_step(self, batch, _):
        """Only used to trigger on_test_epoch_end"""
        self.log(name="loss", value=0.0, prog_bar=True)

    def validation_epoch_end(self, output):
        return None

    def configure_optimizers(self):
        return None
