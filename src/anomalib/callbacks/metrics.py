"""MetricsManager callback."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import asdict
from enum import Enum
from typing import Any

import torch
from torchmetrics import Metric, MetricCollection
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib import TaskType
from anomalib.data import Batch
from anomalib.metrics import AnomalibMetricCollection, create_metric_collection
from anomalib.models import AnomalyModule
from torch.nn import ModuleList

logger = logging.getLogger(__name__)


class Device(str, Enum):
    """Device on which to compute metrics."""

    CPU = "cpu"
    GPU = "gpu"


class _MetricsCallback(Callback):
    """Create image and pixel-level AnomalibMetricsCollection.

    This callback creates AnomalibMetricsCollection based on the
        list of strings provided for image and pixel-level metrics.
    After these MetricCollections are created, the callback assigns
    these to the lightning module.

    Args:
        task (TaskType | str): Task type of the current run.
        image_metrics (list[str] | str | dict[str, dict[str, Any]] | None): List of image-level metrics.
        pixel_metrics (list[str] | str | dict[str, dict[str, Any]] | None): List of pixel-level metrics.
        device (str): Whether to compute metrics on cpu or gpu. Defaults to cpu.
    """

    def __init__(
        self,
        metrics: list[AnomalibMetricCollection],
        compute_on_cpu = True,
    ) -> None:
        super().__init__()
        if compute_on_cpu:
            self.metrics_to_cpu(metrics)
        self.metrics = metrics

    def setup(self, trainer: Trainer, pl_module: AnomalyModule, stage: str) -> None:
        del trainer, stage
        pl_module.metrics = ModuleList(self.metrics)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, outputs, batch_idx, dataloader_idx, pl_module  # Unused arguments.
        for metric in self.metrics:
            metric.update_from_batch(batch)

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: AnomalyModule,
    ) -> None:
        del trainer, pl_module  # Unused argument.
        for metric_collection in self.metrics:
            self.log_dict(metric_collection, prog_bar=True)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, outputs, batch_idx, dataloader_idx, pl_module  # Unused arguments.
        for metric in self.metrics:
            metric.update_from_batch(batch)

    def on_test_epoch_end(
        self,
        trainer: Trainer,
        pl_module: AnomalyModule,
    ) -> None:
        del trainer, pl_module  # Unused argument.
        for metric_collection in self.metrics:
            self.log_dict(metric_collection, prog_bar=True)

    def metrics_to_cpu(self, metrics: Metric | MetricCollection | list[MetricCollection]) -> None:
        if isinstance(metrics, Metric):
            metrics.compute_on_cpu = True
        else:
            metrics = metrics if isinstance(metrics, list) else metrics.values()
            for metric in metrics:
                self.metrics_to_cpu(metric)
