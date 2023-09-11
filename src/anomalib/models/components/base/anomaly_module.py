"""Base Anomaly Module for Training Task."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from abc import ABC
from typing import Any, OrderedDict
from warnings import warn

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
from torchmetrics import Metric

from anomalib.post_processing import ThresholdMethod
from anomalib.utils.metrics import AnomalibMetricCollection, AnomalyScoreDistribution, AnomalyScoreThreshold, MinMax

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

        self.threshold_method: ThresholdMethod
        self.image_threshold = AnomalyScoreThreshold().cpu()
        self.pixel_threshold = AnomalyScoreThreshold().cpu()

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

        if self.threshold_method == ThresholdMethod.ADAPTIVE:
            self._collect_output(self.image_threshold, self.pixel_threshold, outputs)

        self._collect_output(self.image_metrics, self.pixel_metrics, outputs)

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        """Called at the end of each test step."""
        self._outputs_to_cpu(outputs)

        self._collect_output(self.image_metrics, self.pixel_metrics, outputs)

    def on_validation_epoch_start(self):
        if self.threshold_method == ThresholdMethod.ADAPTIVE:
            self.image_threshold.reset()
            self.pixel_threshold.reset()

        self.image_metrics.reset()
        self.pixel_metrics.reset()

    def on_validation_epoch_end(self):
        if self.threshold_method == ThresholdMethod.ADAPTIVE:
            # Calculate thresholds before computing metrics.
            self._compute_adaptive_threshold()

        self._log_metrics()

    def on_test_epoch_start(self):
        self.image_metrics.reset()
        self.pixel_metrics.reset()

    def on_test_epoch_end(self):
        self._log_metrics()

    def _compute_adaptive_threshold(self) -> None:
        self.image_threshold.compute()

        if self.pixel_threshold._update_called:
            self.pixel_threshold.compute()
        else:
            self.pixel_threshold.value = self.image_threshold.value

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

    def _load_normalization_class(self, state_dict: OrderedDict[str, Tensor]) -> None:
        """Assigns the normalization method to use."""
        if "normalization_metrics.max" in state_dict.keys():
            self.normalization_metrics = MinMax()
        elif "normalization_metrics.image_mean" in state_dict.keys():
            self.normalization_metrics = AnomalyScoreDistribution()
        else:
            warn("No known normalization found in model weights.")

    def load_state_dict(self, state_dict: OrderedDict[str, Tensor], strict: bool = True):
        """Load state dict from checkpoint.

        Ensures that normalization and thresholding attributes is properly setup before model is loaded.
        """
        # Used to load missing normalization and threshold parameters
        self._load_normalization_class(state_dict)
        return super().load_state_dict(state_dict, strict=strict)
