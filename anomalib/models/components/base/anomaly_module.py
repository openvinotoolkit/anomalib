"""Base Anomaly Module for Training Task."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC
from typing import Any, List, Optional, OrderedDict
from warnings import warn

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from torch import Tensor, nn
from torchmetrics import Metric

from anomalib.post_processing import ThresholdMethod
from anomalib.utils.metrics import (
    AnomalibMetricCollection,
    AnomalyScoreDistribution,
    AnomalyScoreThreshold,
    MinMax,
)

logger = logging.getLogger(__name__)


class AnomalyModule(pl.LightningModule, ABC):
    """AnomalyModule to train, validate, predict and test images.

    Acts as a base class for all the Anomaly Modules in the library.
    """

    def __init__(self):
        super().__init__()
        logger.info("Initializing %s model.", self.__class__.__name__)

        self.save_hyperparameters()
        self.model: nn.Module
        self.loss: Tensor
        self.callbacks: List[Callback]

        self.threshold_method: ThresholdMethod
        self.image_threshold = AnomalyScoreThreshold().cpu()
        self.pixel_threshold = AnomalyScoreThreshold().cpu()

        self.normalization_metrics: Metric

        self.image_metrics: AnomalibMetricCollection
        self.pixel_metrics: AnomalibMetricCollection

    def forward(self, batch):  # pylint: disable=arguments-differ
        """Forward-pass input tensor to the module.

        Args:
            batch (Tensor): Input Tensor

        Returns:
            Tensor: Output tensor from the model.
        """
        return self.model(batch)

    def validation_step(self, batch, batch_idx) -> dict:  # type: ignore  # pylint: disable=arguments-differ
        """To be implemented in the subclasses."""
        raise NotImplementedError

    def predict_step(self, batch: Any, batch_idx: int, _dataloader_idx: Optional[int] = None) -> Any:
        """Step function called during :meth:`~pytorch_lightning.trainer.trainer.Trainer.predict`.

        By default, it calls :meth:`~pytorch_lightning.core.lightning.LightningModule.forward`.
        Override to add any processing logic.

        Args:
            batch (Tensor): Current batch
            batch_idx (int): Index of current batch
            _dataloader_idx (int): Index of the current dataloader

        Return:
            Predicted output
        """
        outputs = self.validation_step(batch, batch_idx)
        self._post_process(outputs)
        outputs["pred_labels"] = outputs["pred_scores"] >= self.image_threshold.value
        if "anomaly_maps" in outputs.keys():
            outputs["pred_masks"] = outputs["anomaly_maps"] >= self.pixel_threshold.value
        return outputs

    def test_step(self, batch, _):  # pylint: disable=arguments-differ
        """Calls validation_step for anomaly map/score calculation.

        Args:
          batch (Tensor): Input batch
          _: Index of the batch.

        Returns:
          Dictionary containing images, features, true labels and masks.
          These are required in `validation_epoch_end` for feature concatenation.
        """
        return self.predict_step(batch, _)

    def validation_step_end(self, val_step_outputs):  # pylint: disable=arguments-differ
        """Called at the end of each validation step."""
        self._outputs_to_cpu(val_step_outputs)
        self._post_process(val_step_outputs)
        return val_step_outputs

    def test_step_end(self, test_step_outputs):  # pylint: disable=arguments-differ
        """Called at the end of each test step."""
        self._outputs_to_cpu(test_step_outputs)
        self._post_process(test_step_outputs)
        return test_step_outputs

    def validation_epoch_end(self, outputs):
        """Compute threshold and performance metrics.

        Args:
          outputs: Batch of outputs from the validation step
        """
        if self.threshold_method == ThresholdMethod.ADAPTIVE:
            self._compute_adaptive_threshold(outputs)
        self._collect_outputs(self.image_metrics, self.pixel_metrics, outputs)
        self._log_metrics()

    def test_epoch_end(self, outputs):
        """Compute and save anomaly scores of the test set.

        Args:
            outputs: Batch of outputs from the validation step
        """
        self._collect_outputs(self.image_metrics, self.pixel_metrics, outputs)
        self._log_metrics()

    def _compute_adaptive_threshold(self, outputs):
        self.image_threshold.reset()
        self.pixel_threshold.reset()
        self._collect_outputs(self.image_threshold, self.pixel_threshold, outputs)
        self.image_threshold.compute()
        if "mask" in outputs[0].keys() and "anomaly_maps" in outputs[0].keys():
            self.pixel_threshold.compute()
        else:
            self.pixel_threshold.value = self.image_threshold.value

        self.image_metrics.set_threshold(self.image_threshold.value.item())
        self.pixel_metrics.set_threshold(self.pixel_threshold.value.item())

    @staticmethod
    def _collect_outputs(image_metric, pixel_metric, outputs):
        for output in outputs:
            image_metric.cpu()
            image_metric.update(output["pred_scores"], output["label"].int())
            if "mask" in output.keys() and "anomaly_maps" in output.keys():
                pixel_metric.cpu()
                pixel_metric.update(output["anomaly_maps"], output["mask"].int())

    @staticmethod
    def _post_process(outputs):
        """Compute labels based on model predictions."""
        if "pred_scores" not in outputs and "anomaly_maps" in outputs:
            outputs["pred_scores"] = (
                outputs["anomaly_maps"].reshape(outputs["anomaly_maps"].shape[0], -1).max(dim=1).values
            )

    @staticmethod
    def _outputs_to_cpu(output):
        for key, value in output.items():
            if isinstance(value, Tensor):
                output[key] = value.cpu()

    def _log_metrics(self):
        """Log computed performance metrics."""
        if self.pixel_metrics.update_called:
            self.log_dict(self.pixel_metrics, prog_bar=True)
            self.log_dict(self.image_metrics, prog_bar=False)
        else:
            self.log_dict(self.image_metrics, prog_bar=True)

    def _load_normalization_class(self, state_dict: OrderedDict[str, Tensor]):
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
