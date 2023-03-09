"""Metrics Configuration Callback."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from anomalib.data import TaskType
from anomalib.models.components.base.anomaly_module import AnomalyModule
from anomalib.utils.metrics import create_metric_collection

logger = logging.getLogger(__name__)

__all__ = ["MetricsConfigurationCallback"]


class MetricsConfigurationCallback(Callback):
    """Metrics Configuration Callback."""

    def __init__(
        self,
        task: TaskType = TaskType.SEGMENTATION,
        image_metrics: list[str] | None = None,
        pixel_metrics: list[str] | None = None,
    ) -> None:
        """Create image and pixel-level AnomalibMetricsCollection.

        This callback creates AnomalibMetricsCollection based on the
            list of strings provided for image and pixel-level metrics.
        After these MetricCollections are created, the callback assigns
        these to the lightning module.

        Args:
            task (TaskType): Task type of the current run.
            image_metrics (list[str] | None): List of image-level metrics.
            pixel_metrics (list[str] | None): List of pixel-level metrics.
        """
        self.task = task
        self.image_metric_names = image_metrics
        self.pixel_metric_names = pixel_metrics

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: AnomalyModule,
        stage: str | None = None,
    ) -> None:
        """Setup image and pixel-level AnomalibMetricsCollection within Anomalib Model.

        Args:
            trainer (pl.Trainer): PyTorch Lightning Trainer
            pl_module (AnomalyModule): Anomalib Model that inherits pl LightningModule.
            stage (str | None, optional): fit, validate, test or predict. Defaults to None.
        """
        del trainer, stage  # These variables are not used.

        image_metric_names = [] if self.image_metric_names is None else self.image_metric_names

        pixel_metric_names: list[str]
        if self.pixel_metric_names is None:
            pixel_metric_names = []
        elif self.task == TaskType.CLASSIFICATION:
            pixel_metric_names = []
            logger.warning(
                "Cannot perform pixel-level evaluation when task type is classification. "
                "Ignoring the following pixel-level metrics: %s",
                self.pixel_metric_names,
            )
        else:
            pixel_metric_names = self.pixel_metric_names

        if isinstance(pl_module, AnomalyModule):
            pl_module.image_metrics = create_metric_collection(image_metric_names, "image_")
            pl_module.pixel_metrics = create_metric_collection(pixel_metric_names, "pixel_")

            pl_module.image_metrics.set_threshold(pl_module.image_threshold.value)
            pl_module.pixel_metrics.set_threshold(pl_module.pixel_threshold.value)
