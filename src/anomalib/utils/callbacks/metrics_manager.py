"""MetricsManager callback."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging
from typing import Any

import torch
from lightning.pytorch import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor

from anomalib import trainer
from anomalib.data import TaskType
from anomalib.models import AnomalyModule
from anomalib.utils.metrics import AnomalibMetricCollection, create_metric_collection

logger = logging.getLogger(__name__)


class _MetricsManagerCallback(Callback):
    """Create image and pixel-level AnomalibMetricsCollection.


    Note: This callback is set within the AnomalibTrainer.

    This callback creates AnomalibMetricsCollection based on the
        list of strings provided for image and pixel-level metrics.
    After these MetricCollections are created, the callback assigns
    these to the lightning module.

    Args:
        task (TaskType): Task type of the current run.
        image_metrics (list[str] | None): List of image-level metrics.
        pixel_metrics (list[str] | None): List of pixel-level metrics.
    """

    def __init__(
        self,
        task: TaskType = TaskType.SEGMENTATION,
        image_metrics: list[str] | None = None,
        pixel_metrics: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.task = task
        self.image_metric_names = image_metrics
        self.pixel_metric_names = pixel_metrics

    def setup(
        self,
        trainer: "trainer.AnomalibTrainer",
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

    def on_validation_epoch_start(
        self,
        trainer: "trainer.AnomalibTrainer",
        pl_module: AnomalyModule,
    ) -> None:
        pl_module.image_metrics.reset()
        pl_module.pixel_metrics.reset()

    def on_validation_batch_end(
        self,
        trainer: "trainer.AnomalibTrainer",
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if outputs is not None:
            self._outputs_to_cpu(outputs)
            self._collect_output(pl_module.image_metrics, pl_module.pixel_metrics, outputs)

    def on_validation_epoch_end(
        self,
        trainer: "trainer.AnomalibTrainer",
        pl_module: AnomalyModule,
    ) -> None:
        self._update_metrics_threshold(pl_module)
        self._log_metrics(pl_module)

    def on_test_epoch_start(
        self,
        trainer: "trainer.AnomalibTrainer",
        pl_module: AnomalyModule,
    ) -> None:
        pl_module.image_metrics.reset()
        pl_module.pixel_metrics.reset()

    def on_test_batch_end(
        self,
        trainer: "trainer.AnomalibTrainer",
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if outputs is not None:
            self._outputs_to_cpu(outputs)
            self._collect_output(pl_module.image_metrics, pl_module.pixel_metrics, outputs)

    def on_test_epoch_end(
        self,
        trainer: "trainer.AnomalibTrainer",
        pl_module: AnomalyModule,
    ) -> None:
        self._log_metrics(pl_module)

    def _update_metrics_threshold(self, pl_module: AnomalyModule) -> None:
        pl_module.image_metrics.set_threshold(pl_module.image_threshold.value.item())
        pl_module.pixel_metrics.set_threshold(pl_module.pixel_threshold.value.item())

    @staticmethod
    def _collect_output(
        image_metric: AnomalibMetricCollection,
        pixel_metric: AnomalibMetricCollection,
        output: STEP_OUTPUT,
    ) -> None:
        image_metric.cpu()
        image_metric.update(output["pred_scores"], output["label"].int())
        if "mask" in output.keys() and "anomaly_maps" in output.keys():
            pixel_metric.cpu()
            pixel_metric.update(torch.squeeze(output["anomaly_maps"]), torch.squeeze(output["mask"].int()))

    def _outputs_to_cpu(self, output: STEP_OUTPUT):
        if isinstance(output, dict):
            for key, value in output.items():
                output[key] = self._outputs_to_cpu(value)
        elif isinstance(output, Tensor):
            output = output.cpu()
        return output

    @staticmethod
    def _log_metrics(pl_module: AnomalyModule) -> None:
        """Log computed performance metrics."""
        if pl_module.pixel_metrics._update_called:
            pl_module.log_dict(pl_module.pixel_metrics, prog_bar=True)
            pl_module.log_dict(pl_module.image_metrics, prog_bar=False)
        else:
            pl_module.log_dict(pl_module.image_metrics, prog_bar=True)
