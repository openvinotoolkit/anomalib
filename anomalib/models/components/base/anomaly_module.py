"""Base Anomaly Module for Training Task."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC
from importlib import import_module
from typing import Any, List, Optional, Union

import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks.base import Callback
from torch import Tensor, nn
from torchmetrics import Metric

from anomalib.utils.metrics import AnomalibMetricCollection, BaseThreshold

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

        # Defaults to "adaptive thresholding"
        self._threshold_type: str
        self._image_threshold: BaseThreshold
        self._pixel_threshold: BaseThreshold

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
        if self._threshold_type == "adaptive":
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
        self._collect_outputs(self.image_threshold, self.pixel_threshold, outputs)
        self.image_threshold.compute()
        if "mask" in outputs[0].keys() and "anomaly_maps" in outputs[0].keys():
            self.pixel_threshold.compute()
        else:
            self.pixel_threshold.value = self.image_threshold.value

        self.image_metrics.set_threshold(self.image_threshold.value.item())
        self.pixel_metrics.set_threshold(self.pixel_threshold.value.item())

    def _collect_outputs(self, image_metric, pixel_metric, outputs):
        for output in outputs:
            image_metric.cpu()
            image_metric.update(output["pred_scores"], output["label"].int())
            if "mask" in output.keys() and "anomaly_maps" in output.keys():
                pixel_metric.cpu()
                pixel_metric.update(output["anomaly_maps"], output["mask"].int())

    def _post_process(self, outputs):
        """Compute labels based on model predictions."""
        if "pred_scores" not in outputs and "anomaly_maps" in outputs:
            outputs["pred_scores"] = (
                outputs["anomaly_maps"].reshape(outputs["anomaly_maps"].shape[0], -1).max(dim=1).values
            )

    def _outputs_to_cpu(self, output):
        # for output in outputs:
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

    @property
    def image_threshold(self):
        """Get the image threshold."""
        if not hasattr(self, "_image_threshold"):
            raise AttributeError("Image threshold is not initialized.")
        return self._image_threshold

    @property
    def pixel_threshold(self):
        """Get the pixel threshold."""
        if not hasattr(self, "_pixel_threshold"):
            raise AttributeError("Pixel threshold is not initialized")
        return self._pixel_threshold

    @property
    def threshold(self) -> str:
        """Returns the name of the threshold type."""
        return self._threshold_type

    @threshold.setter
    def threshold(self, value: Union[str, DictConfig]) -> None:
        """Used to assign threshold by using assignment operator.

        Args:
            value (Union[str, DictConfig]): Threshold type or config dict.

        Raises:
            ValueError: If the threshold type is not supported.
        """
        if not (
            hasattr(self, "_image_threshold") and hasattr(self, "_pixel_threshold")
        ):  # Set threshold only if it is not initialized
            if isinstance(value, str):
                self._threshold_type = value
                thresholding_args = {}
            else:
                assert len(list(value.keys())) == 1, (
                    "threshold should either be a string or a dictionary with one thresholding type."
                    " with one key. See `https://openvinotoolkit.github.io/anomalib/guides/thresholding.html`"
                    " for more details"
                )
                self._threshold_type = str(list(value.keys())[0])
                thresholding_args = value[self._threshold_type]

            # check for valid thresholding method.
            if self._threshold_type not in ["adaptive", "manual", "maximum"]:
                raise ValueError(f"Invalid threshold type: {self._threshold_type}")

            # copy devault_value key to both image and pixel threshold and remove it from thresholding_args
            if (
                "default_value" in thresholding_args
            ):  # if default value is provided, use it for image and pixel threshold.
                thresholding_args["image_threshold"] = thresholding_args["default_value"]
                thresholding_args["pixel_threshold"] = thresholding_args["default_value"]
                thresholding_args.pop("default_value")
            if self._threshold_type == "manual":
                assert (
                    "image_threshold" in thresholding_args or "pixel_threshold" in thresholding_args
                ), "Need to provide image_threshold or pixel_threshold"
                if "image_threshold" not in thresholding_args:
                    thresholding_args["image_threshold"] = thresholding_args["pixel_threshold"]
                elif "pixel_threshold" not in thresholding_args:
                    thresholding_args["pixel_threshold"] = thresholding_args["image_threshold"]

            self._image_threshold = self._assign_threshold_class(
                self._threshold_type, thresholding_args.get("image_threshold", 0.0), **thresholding_args
            )
            self._pixel_threshold = self._assign_threshold_class(
                self._threshold_type, thresholding_args.get("pixel_threshold", 0.0), **thresholding_args
            )

    def _assign_threshold_class(self, class_name: str, default_value: float, **thresholding_args: Any) -> BaseThreshold:
        """Assign class to image threshold and pixel threshold.

        Args:
            class_name (str): name of the thresholding class
            default_value (float): default value of the threshold
            thresholding_args (Any): arguments for the thresholding class

        Returns:
            BaseThreshold: threshold class
        """
        # importlib is used as mypy throws type error for BaseThreshold
        module = import_module("anomalib.utils.metrics.thresholding")
        threshold_class = getattr(module, class_name.capitalize() + "Threshold")
        return threshold_class(default_value, **thresholding_args).cpu()
