"""Anomaly Score Normalization Callback that uses min-max normalization."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from typing import Any

import torch
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib.metrics import MinMax
from anomalib.models.components import AnomalyModule
from anomalib.utils.normalization.min_max import normalize


class _MinMaxNormalizationCallback(Callback):
    """Callback that normalizes the image-level and pixel-level anomaly scores using min-max normalization.

    Note: This callback is set within the Engine.
    """

    def setup(self, trainer: Trainer, pl_module: AnomalyModule, stage: str | None = None) -> None:
        """Add min_max metrics to normalization metrics."""
        del trainer, stage  # These variables are not used.

        if not hasattr(pl_module, "normalization_metrics"):
            pl_module.normalization_metrics = MinMax().cpu()
        elif not isinstance(pl_module.normalization_metrics, MinMax):
            msg = f"Expected normalization_metrics to be of type MinMax, got {type(pl_module.normalization_metrics)}"
            raise AttributeError(
                msg,
            )

    def on_test_start(self, trainer: Trainer, pl_module: AnomalyModule) -> None:
        """Call when the test begins."""
        del trainer  # `trainer` variable is not used.

        for metric in (pl_module.image_metrics, pl_module.pixel_metrics):
            if metric is not None:
                metric.set_threshold(0.5)

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

        if "anomaly_maps" in outputs:
            pl_module.normalization_metrics(outputs["anomaly_maps"])
        elif "box_scores" in outputs:
            pl_module.normalization_metrics(torch.cat(outputs["box_scores"]))
        elif "pred_scores" in outputs:
            pl_module.normalization_metrics(outputs["pred_scores"])
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
        if "pred_scores" in outputs:
            outputs["pred_scores"] = normalize(outputs["pred_scores"], image_threshold, stats.min, stats.max)
        if "anomaly_maps" in outputs:
            outputs["anomaly_maps"] = normalize(outputs["anomaly_maps"], pixel_threshold, stats.min, stats.max)
        if "box_scores" in outputs:
            outputs["box_scores"] = [
                normalize(scores, pixel_threshold, stats.min, stats.max) for scores in outputs["box_scores"]
            ]
