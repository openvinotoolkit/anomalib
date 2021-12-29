"""Anomaly Score Normalization Callback that uses min-max normalization."""
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT


class MinMaxNormalizationCallback(Callback):
    """Callback that normalizes the image-level and pixel-level anomaly scores using min-max normalization."""

    def __init__(self):
        self.min = float("inf")
        self.max = -float("inf")

    def on_test_start(self, _trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when the test begins."""
        normalized_image_threshold = (pl_module.image_threshold.value.item() - self.min) / (self.max - self.min)
        normalized_pixel_threshold = (pl_module.pixel_threshold.value.item() - self.min) / (self.max - self.min)
        pl_module.image_metrics.F1.threshold = normalized_image_threshold
        pl_module.pixel_metrics.F1.threshold = normalized_pixel_threshold

    def on_validation_batch_end(
        self,
        _trainer: pl.Trainer,
        _pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Called when the validation batch ends, update the min and max observed values."""
        batch_max = torch.max(outputs["anomaly_maps"]).item()
        batch_min = torch.min(outputs["anomaly_maps"]).item()
        self.max = max(self.max, batch_max)
        self.min = min(self.min, batch_min)

    def on_test_batch_end(
        self,
        _trainer: pl.Trainer,
        _pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Called when the test batch ends, normalizes the predicted scores and anomaly maps."""
        self._normalize(outputs)

    def on_predict_batch_end(
        self,
        _trainer: pl.Trainer,
        _pl_module: pl.LightningModule,
        outputs: Dict,
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Called when the predict batch ends, normalizes the predicted scores and anomaly maps."""
        self._normalize(outputs)

    def _normalize(self, outputs):
        outputs["pred_scores"] = (outputs["pred_scores"] - self.min) / (self.max - self.min)
        outputs["anomaly_maps"] = (outputs["anomaly_maps"] - self.min) / (self.max - self.min)
