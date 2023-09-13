"""Base Anomaly Module for Training Task."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from abc import ABC
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
from torchmetrics import Metric

from anomalib.utils.metrics import AnomalibMetricCollection
from anomalib.utils.metrics.threshold import BaseThreshold

logger = logging.getLogger(__name__)


class AnomalyModule(pl.LightningModule, ABC):
    """AnomalyModule to train, validate, predict and test images.

    Acts as a base class for all the Anomaly Modules in the library.
    """

    def __init__(self) -> None:
        super().__init__()
        logger.info("Initializing %s model.", self.__class__.__name__)

        self.save_hyperparameters()
        self.model: nn.Module
        self.loss: nn.Module
        self.callbacks: list[Callback]

        self.image_threshold: BaseThreshold
        self.pixel_threshold: BaseThreshold

        self.normalization_metrics: Metric

        self.image_metrics: AnomalibMetricCollection
        self.pixel_metrics: AnomalibMetricCollection

    def forward(self, batch: dict[str, str | Tensor], *args, **kwargs) -> Any:
        """Forward-pass input tensor to the module.

        Args:
            batch (dict[str, str | Tensor]): Input batch.

        Returns:
            Tensor: Output tensor from the model.
        """
        del args, kwargs  # These variables are not used.

        return self.model(batch)

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """To be implemented in the subclasses."""
        raise NotImplementedError

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """Step function called during :meth:`~lightning.pytorch.trainer.trainer.Trainer.predict`.

        By default, it calls :meth:`~lightning.pytorch.core.lightning.LightningModule.forward`.
        Override to add any processing logic.

        Args:
            batch (Any): Current batch
            batch_idx (int): Index of current batch
            dataloader_idx (int): Index of the current dataloader

        Return:
            Predicted output
        """
        del batch_idx, dataloader_idx  # These variables are not used.

        return self.validation_step(batch)

    def test_step(self, batch: dict[str, str | Tensor], batch_idx: int, *args, **kwargs) -> STEP_OUTPUT:
        """Calls validation_step for anomaly map/score calculation.

        Args:
          batch (dict[str, str | Tensor]): Input batch
          batch_idx (int): Batch index

        Returns:
          Dictionary containing images, features, true labels and masks.
          These are required in `validation_epoch_end` for feature concatenation.
        """
        del args, kwargs  # These variables are not used.

        return self.predict_step(batch, batch_idx)

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        """Called at the end of each validation step."""
        self._outputs_to_cpu(outputs)

        self._collect_output(self.image_metrics, self.pixel_metrics, outputs)

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        """Called at the end of each test step."""
        self._outputs_to_cpu(outputs)

        self._collect_output(self.image_metrics, self.pixel_metrics, outputs)

    def on_validation_epoch_start(self):
        self.image_metrics.reset()
        self.pixel_metrics.reset()

    def on_validation_epoch_end(self):
        self._update_metrics_threshold()

        self._log_metrics()

    def on_test_epoch_start(self):
        self.image_metrics.reset()
        self.pixel_metrics.reset()

    def on_test_epoch_end(self):
        self._log_metrics()

    def _update_metrics_threshold(self) -> None:
        self.image_metrics.set_threshold(self.image_threshold.value.item())
        self.pixel_metrics.set_threshold(self.pixel_threshold.value.item())

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

    def _outputs_to_cpu(self, output):
        if isinstance(output, dict):
            for key, value in output.items():
                output[key] = self._outputs_to_cpu(value)
        elif isinstance(output, list):
            output = [self._outputs_to_cpu(item) for item in output]
        elif isinstance(output, Tensor):
            output = output.cpu()
        return output

    def _log_metrics(self) -> None:
        """Log computed performance metrics."""
        if self.pixel_metrics._update_called:
            self.log_dict(self.pixel_metrics, prog_bar=True)
            self.log_dict(self.image_metrics, prog_bar=False)
        else:
            self.log_dict(self.image_metrics, prog_bar=True)
