"""Anomaly Score Normalization Callback that uses min-max normalization."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import MetricCollection

from anomalib.metrics import MinMax
from anomalib.models.components import AnomalyModule
from anomalib.utils.normalization.min_max import normalize

from .base import NormalizationCallback


class _MinMaxNormalizationCallback(NormalizationCallback):
    """Callback that normalizes the image-level and pixel-level anomaly scores using min-max normalization.

    Note: This callback is set within the Engine.
    """

    @staticmethod
    def setup(trainer: Trainer, pl_module: AnomalyModule, stage: str | None = None) -> None:
        """Add min_max metrics to normalization metrics."""
        del trainer, stage  # These variables are not used.

        if not hasattr(pl_module, "normalization_metrics"):
            pl_module.normalization_metrics = MetricCollection(
                {
                    "anomaly_maps": MinMax().cpu(),
                    "box_scores": MinMax().cpu(),
                    "pred_scores": MinMax().cpu(),
                },
            )

        elif not isinstance(pl_module.normalization_metrics, MetricCollection):
            msg = (
                f"Expected normalization_metrics to be of type MetricCollection"
                f"got {type(pl_module.normalization_metrics)}"
            )
            raise TypeError(msg)

        for name, metric in pl_module.normalization_metrics.items():
            if not isinstance(metric, MinMax):
                msg = f"Expected normalization_metric {name} to be of type MinMax, got {type(metric)}"
                raise TypeError(msg)

    @staticmethod
    def on_test_start(trainer: Trainer, pl_module: AnomalyModule) -> None:
        """Call when the test begins."""
        del trainer  # `trainer` variable is not used.

        for metric in (pl_module.image_metrics, pl_module.pixel_metrics):
            if metric is not None:
                metric.set_threshold(0.5)

    @staticmethod
    def on_validation_epoch_start(trainer: Trainer, pl_module: AnomalyModule) -> None:
        """Call when the validation epoch begins."""
        del trainer  # `trainer` variable is not used.

        if hasattr(pl_module, "normalization_metrics"):
            pl_module.normalization_metrics.reset()

    @staticmethod
    def on_validation_batch_end(
        trainer: Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Call when the validation batch ends, update the min and max observed values."""
        del trainer, batch, batch_idx, dataloader_idx  # These variables are not used.

        if "anomaly_maps" in outputs:
            pl_module.normalization_metrics["anomaly_maps"](outputs["anomaly_maps"])
        if "box_scores" in outputs:
            pl_module.normalization_metrics["box_scores"](torch.cat(outputs["box_scores"]))
        if "pred_scores" in outputs:
            pl_module.normalization_metrics["pred_scores"](outputs["pred_scores"])

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Call when the test batch ends, normalizes the predicted scores and anomaly maps."""
        del trainer, batch, batch_idx, dataloader_idx  # These variables are not used.

        self._normalize_batch(outputs, pl_module)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: AnomalyModule,
        outputs: Any,  # noqa: ANN401
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Call when the predict batch ends, normalizes the predicted scores and anomaly maps."""
        del trainer, batch, batch_idx, dataloader_idx  # These variables are not used.

        self._normalize_batch(outputs, pl_module)

    @staticmethod
    def _normalize_batch(outputs: Any, pl_module: AnomalyModule) -> None:  # noqa: ANN401
        """Normalize a batch of predictions."""
        image_threshold = pl_module.image_threshold.value.cpu()
        pixel_threshold = pl_module.pixel_threshold.value.cpu()
        if "pred_scores" in outputs:
            stats = pl_module.normalization_metrics["pred_scores"].cpu()
            outputs["pred_scores"] = normalize(outputs["pred_scores"], image_threshold, stats.min, stats.max)
        if "anomaly_maps" in outputs:
            stats = pl_module.normalization_metrics["anomaly_maps"].cpu()
            outputs["anomaly_maps"] = normalize(outputs["anomaly_maps"], pixel_threshold, stats.min, stats.max)
        if "box_scores" in outputs:
            stats = pl_module.normalization_metrics["box_scores"].cpu()
            outputs["box_scores"] = [
                normalize(scores, pixel_threshold, stats.min, stats.max) for scores in outputs["box_scores"]
            ]
