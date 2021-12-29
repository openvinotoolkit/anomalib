"""Anomaly Score Normalization Callback that uses min-max normalization."""
from typing import Any, Dict

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT


class MinMaxNormalizationCallback(Callback):
    """Callback that normalizes the image-level and pixel-level anomaly scores using min-max normalization."""

    def on_test_start(self, _trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when the test begins."""
        normalized_image_threshold = (pl_module.image_threshold.value.item() - pl_module.min_max.min) / (
            pl_module.min_max.max - pl_module.min_max.min
        )
        normalized_pixel_threshold = (pl_module.pixel_threshold.value.item() - pl_module.min_max.min) / (
            pl_module.min_max.max - pl_module.min_max.min
        )
        pl_module.image_metrics.F1.threshold = normalized_image_threshold.item()
        pl_module.pixel_metrics.F1.threshold = normalized_pixel_threshold.item()

    def on_validation_batch_end(
        self,
        _trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Called when the validation batch ends, update the min and max observed values."""
        if "anomaly_maps" in outputs.keys():
            pl_module.min_max(outputs["anomaly_maps"])
        else:
            pl_module.min_max(outputs["pred_scores"])

    def on_test_batch_end(
        self,
        _trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Called when the test batch ends, normalizes the predicted scores and anomaly maps."""
        self._normalize(outputs, pl_module.min_max)

    def on_predict_batch_end(
        self,
        _trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict,
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Called when the predict batch ends, normalizes the predicted scores and anomaly maps."""
        self._normalize(outputs, pl_module.min_max)

    def _normalize(self, outputs, stats):
        outputs["pred_scores"] = (outputs["pred_scores"] - stats.min) / (stats.max - stats.min)
        if "anomaly_maps" in outputs.keys():
            outputs["anomaly_maps"] = (outputs["anomaly_maps"] - stats.min) / (stats.max - stats.min)
