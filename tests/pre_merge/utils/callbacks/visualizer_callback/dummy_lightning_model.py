from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from anomalib.models.components import AnomalyModule
from anomalib.utils.callbacks.visualizer_callback import VisualizerCallback


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


class DummyAnomalyMapGenerator:
    def __init__(self):
        self.input_size = (100, 100)
        self.sigma = 4


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anomaly_map_generator = DummyAnomalyMapGenerator()


class DummyModule(AnomalyModule):
    """A dummy module which calls visualizer callback on fake images and masks."""

    def __init__(
        self,
        adaptive_threshold: bool,
        default_image_threshold: float,
        default_pixel_threshold: float,
        normalization: Optional[str] = None,
    ):
        super().__init__(
            adaptive_threshold=adaptive_threshold,
            default_image_threshold=default_image_threshold,
            default_pixel_threshold=default_pixel_threshold,
            normalization=normalization,
        )
        self.model = DummyModel()
        self.task = "segmentation"
        self.callbacks = [VisualizerCallback(task=self.task)]  # test if this is removed

    def test_step(self, batch, _):
        """Only used to trigger on_test_epoch_end."""
        self.log(name="loss", value=0.0, prog_bar=True)
        outputs = dict(
            image_path=[Path("test1.jpg")],
            image=torch.rand((1, 3, 100, 100)),
            mask=torch.zeros((1, 100, 100)),
            anomaly_maps=torch.ones((1, 100, 100)),
            label=torch.Tensor([0]),
        )
        return outputs

    def validation_epoch_end(self, output):
        return None

    def test_epoch_end(self, outputs):
        return None

    def configure_optimizers(self):
        return None
