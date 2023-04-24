from pathlib import Path
from typing import Union

import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from anomalib.models.components import AnomalyModule
from tests.helpers.dataset import get_dataset_path
from tests.helpers.metrics import get_metrics


class DummyModule(AnomalyModule):
    """A dummy model which calls visualizer callback on fake images and masks."""

    def __init__(self, hparams: Union[DictConfig, ListConfig]):
        super().__init__()
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

    def test_dataloader(self):
        """Needed to run test_step."""
        return torch.tensor([0])
