from pathlib import Path
from typing import Union

import pytorch_lightning as pl
import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch import nn
from torch.utils.data import DataLoader, Dataset

from anomalib.models.components import AnomalyModule
from anomalib.utils.callbacks import ImageVisualizerCallback
from anomalib.utils.metrics import get_metrics
from tests.helpers.dataset import get_dataset_path


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


class DummyAnomalyMapGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = (100, 100)
        self.sigma = 4


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anomaly_map_generator = DummyAnomalyMapGenerator()


class DummyModule(AnomalyModule):
    """A dummy model which calls visualizer callback on fake images and masks."""

    def __init__(self, hparams: Union[DictConfig, ListConfig]):
        super().__init__()
        self.model = DummyModel()
        self.task = "segmentation"
        self.mode = "full"
        self.callbacks = [
            ImageVisualizerCallback(
                task=self.task,
                mode=self.mode,
                image_save_path=hparams.project.path + "/images",
                log_images=True,
                save_images=True,
            )
        ]  # test if this is removed

        self.image_metrics, self.pixel_metrics = get_metrics(hparams)
        self.image_metrics.set_threshold(hparams.model.threshold.image_default)
        self.pixel_metrics.set_threshold(hparams.model.threshold.pixel_default)

    def test_step(self, batch, _):
        """Only used to trigger on_test_epoch_end."""
        self.log(name="loss", value=0.0, prog_bar=True)
        outputs = dict(
            image_path=[Path(get_dataset_path("bottle")) / "broken_large/000.png"],
            image=torch.rand((1, 3, 100, 100)),
            mask=torch.zeros((1, 100, 100)),
            anomaly_maps=torch.ones((1, 100, 100)),
            label=torch.Tensor([0]),
            pred_labels=torch.Tensor([0]),
            pred_masks=torch.zeros((1, 100, 100)),
        )
        return outputs

    def validation_epoch_end(self, output):
        return None

    def test_epoch_end(self, outputs):
        return None

    def configure_optimizers(self):
        return None
