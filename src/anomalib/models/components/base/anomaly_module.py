"""Base Anomaly Module for Training Task."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
import logging
from abc import ABC, abstractproperty
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn

from anomalib import LearningType
from anomalib.metrics.threshold import BaseThreshold

if TYPE_CHECKING:
    from lightning.pytorch.callbacks import Callback
    from torchmetrics import Metric

    from anomalib.metrics import AnomalibMetricCollection

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

    def forward(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> Any:  # noqa: ANN401
        """Perform the forward-pass by passing input tensor to the module.

        Args:
            batch (dict[str, str | torch.Tensor]): Input batch.
            *args: Arguments.
            **kwargs: Keyword arguments.

        Returns:
            Tensor: Output tensor from the model.
        """
        del args, kwargs  # These variables are not used.

        return self.model(batch)

    def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """To be implemented in the subclasses."""
        raise NotImplementedError

    def predict_step(
        self,
        batch: dict[str, str | torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> STEP_OUTPUT:
        """Step function called during :meth:`~lightning.pytorch.trainer.Trainer.predict`.

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

    def test_step(self, batch: dict[str, str | torch.Tensor], batch_idx: int, *args, **kwargs) -> STEP_OUTPUT:
        """Calls validation_step for anomaly map/score calculation.

        Args:
          batch (dict[str, str | torch.Tensor]): Input batch
          batch_idx (int): Batch index
          args: Arguments.
          kwargs: Keyword arguments.

        Returns:
          Dictionary containing images, features, true labels and masks.
          These are required in `validation_epoch_end` for feature concatenation.
        """
        del args, kwargs  # These variables are not used.

        return self.predict_step(batch, batch_idx)

    @abstractproperty
    def trainer_arguments(self) -> dict[str, Any]:
        """Arguments used to override the trainer parameters so as to train the model correctly."""
        raise NotImplementedError

    def _save_to_state_dict(self, destination: OrderedDict, prefix: str, keep_vars: bool) -> None:
        destination[
            "image_threshold_class"
        ] = f"{self.image_threshold.__class__.__module__}.{self.image_threshold.__class__.__name__}"
        destination[
            "pixel_threshold_class"
        ] = f"{self.pixel_threshold.__class__.__module__}.{self.pixel_threshold.__class__.__name__}"
        if hasattr(self, "normalization_metrics"):
            normalization_class = self.normalization_metrics.__class__
            destination["normalization_class"] = f"{normalization_class.__module__}.{normalization_class.__name__}"

        return super()._save_to_state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict: OrderedDict[str, Any], strict: bool = True) -> Any:  # noqa: ANN401
        """Initialize auxiliary object."""
        if "image_threshold_class" in state_dict:
            self.image_threshold = self._get_instance(state_dict, "image_threshold_class")
        if "pixel_threshold_class" in state_dict:
            self.pixel_threshold = self._get_instance(state_dict, "pixel_threshold_class")
        if "normalization_class" in state_dict:
            self.normalization_metrics = self._get_instance(state_dict, "normalization_class")

        return super().load_state_dict(state_dict, strict)

    def _get_instance(self, state_dict: OrderedDict[str, Any], dict_key: str) -> BaseThreshold:
        """Get the threshold class from the ``state_dict``."""
        class_path = state_dict.pop(dict_key)
        module = importlib.import_module(".".join(class_path.split(".")[:-1]))
        return getattr(module, class_path.split(".")[-1])()

    @abstractproperty
    def learning_type(self) -> LearningType:
        """Learning type of the model."""
        raise NotImplementedError
