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

    def setup(self, trainer: Trainer, pl_module: AnomalyModule, stage: str | None = None) -> None:
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

    def on_test_start(self, trainer: Trainer, pl_module: AnomalyModule) -> None:
        """Call when the test begins."""
        del trainer  # `trainer` variable is not used.

        for metric in (pl_module.image_metrics, pl_module.pixel_metrics):
            if metric is not None:
                metric.set_threshold(0.5)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: AnomalyModule) -> None:
        """Call when the validation epoch begins."""
        del trainer  # `trainer` variable is not used.

        if hasattr(pl_module, "normalization_metrics"):
            pl_module.normalization_metrics.reset()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Call when the validation batch ends, update the min and max observed values."""
        del trainer, batch, batch_idx, dataloader_idx  # These variables are not used.

        if outputs.anomaly_map is not None:
            pl_module.normalization_metrics(outputs.anomaly_map)
        elif outputs.box_scores is not None:
            pl_module.normalization_metrics(torch.cat(outputs.box_scores))
        elif outputs.pred_score is not None:
            pl_module.normalization_metrics(outputs.pred_score)
        else:
            msg = "No values found for normalization, provide anomaly maps, bbox scores, or image scores"
            raise ValueError(msg)

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
        stats = pl_module.normalization_metrics.cpu()
        if outputs.pred_score is not None:
            outputs.pred_score = normalize(outputs.pred_score, image_threshold, stats.min, stats.max)
        if outputs.anomaly_map is not None:
            outputs.anomaly_map = normalize(outputs.anomaly_map, pixel_threshold, stats.min, stats.max)
        if outputs.box_scores is not None:
            outputs.box_scores = [
                normalize(scores, pixel_threshold, stats.min, stats.max) for scores in outputs.box_scores
            ]
