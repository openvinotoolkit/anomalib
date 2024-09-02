"""MetricsManager callback."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from enum import Enum
from typing import Any

import torch
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib import TaskType
from anomalib.metrics import AnomalibMetricCollection, create_metric_collection
from anomalib.models import AnomalyModule

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
        task: TaskType | str = TaskType.SEGMENTATION,
        image_metrics: list[str] | str | dict[str, dict[str, Any]] | None = None,
        pixel_metrics: list[str] | str | dict[str, dict[str, Any]] | None = None,
        device: Device = Device.CPU,
    ) -> None:
        super().__init__()
        self.task = TaskType(task)
        self.image_metric_names = image_metrics
        self.pixel_metric_names = pixel_metrics
        self.device = device

    def setup(
        self,
        trainer: Trainer,
        pl_module: AnomalyModule,
        stage: str | None = None,
    ) -> None:
        """Set image and pixel-level AnomalibMetricsCollection within Anomalib Model.

        Args:
            trainer (pl.Trainer): PyTorch Lightning Trainer
            pl_module (AnomalyModule): Anomalib Model that inherits pl LightningModule.
            stage (str | None, optional): fit, validate, test or predict. Defaults to None.
        """
        del trainer, stage  # These variables are not used.

        image_metric_names = [] if self.image_metric_names is None else self.image_metric_names
        if isinstance(image_metric_names, str):
            image_metric_names = [image_metric_names]

        pixel_metric_names: list[str] | dict[str, dict[str, Any]]
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
            pixel_metric_names = (
                self.pixel_metric_names if not isinstance(self.pixel_metric_names, str) else [self.pixel_metric_names]
            )

        if isinstance(pl_module, AnomalyModule):
            pl_module.image_metrics = create_metric_collection(image_metric_names, "image_")
            if hasattr(pl_module, "pixel_metrics"):  # incase metrics are loaded from model checkpoint
                new_metrics = create_metric_collection(pixel_metric_names)
                for name in new_metrics:
                    if name not in pl_module.pixel_metrics:
                        pl_module.pixel_metrics.add_metrics(new_metrics[name])
            else:
                pl_module.pixel_metrics = create_metric_collection(pixel_metric_names, "pixel_")
            self._set_threshold(pl_module)

    @staticmethod
    def on_validation_epoch_start(trainer: Trainer, pl_module: AnomalyModule) -> None:
        del trainer  # Unused argument.

        pl_module.image_metrics.reset()
        pl_module.pixel_metrics.reset()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, batch, batch_idx, dataloader_idx  # Unused arguments.

        if outputs is not None:
            self._outputs_to_device(outputs)
            self._update_metrics(pl_module.image_metrics, pl_module.pixel_metrics, outputs)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: AnomalyModule) -> None:
        del trainer  # Unused argument.

        self._set_threshold(pl_module)
        self._log_metrics(pl_module)

    @staticmethod
    def on_test_epoch_start(trainer: Trainer, pl_module: AnomalyModule) -> None:
        del trainer  # Unused argument.

        pl_module.image_metrics.reset()
        pl_module.pixel_metrics.reset()

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, batch, batch_idx, dataloader_idx  # Unused arguments.

        if outputs is not None:
            self._outputs_to_device(outputs)
            self._update_metrics(pl_module.image_metrics, pl_module.pixel_metrics, outputs)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: AnomalyModule) -> None:
        del trainer  # Unused argument.

        self._log_metrics(pl_module)

    @staticmethod
    def _set_threshold(pl_module: AnomalyModule) -> None:
        pl_module.image_metrics.set_threshold(pl_module.image_threshold.value.item())
        pl_module.pixel_metrics.set_threshold(pl_module.pixel_threshold.value.item())

    def _update_metrics(
        self,
        image_metric: AnomalibMetricCollection,
        pixel_metric: AnomalibMetricCollection,
        output: STEP_OUTPUT,
    ) -> None:
        image_metric.to(self.device)
        image_metric.update(output["pred_scores"], output["label"].int())
        if "mask" in output and "anomaly_maps" in output:
            pixel_metric.to(self.device)
            pixel_metric.update(torch.squeeze(output["anomaly_maps"]), torch.squeeze(output["mask"].int()))

    def _outputs_to_device(self, output: STEP_OUTPUT) -> STEP_OUTPUT | dict[str, Any]:
        if isinstance(output, dict):
            for key, value in output.items():
                output[key] = self._outputs_to_device(value)
        elif isinstance(output, torch.Tensor):
            output = output.to(self.device)
        return output

    @staticmethod
    def _log_metrics(pl_module: AnomalyModule) -> None:
        """Log computed performance metrics."""
        if pl_module.pixel_metrics._update_called:  # noqa: SLF001
            pl_module.log_dict(pl_module.pixel_metrics, prog_bar=True)
            pl_module.log_dict(pl_module.image_metrics, prog_bar=False)
        else:
            pl_module.log_dict(pl_module.image_metrics, prog_bar=True)
