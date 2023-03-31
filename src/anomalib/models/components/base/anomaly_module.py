"""Base Anomaly Module for Training Task."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from abc import ABC
from typing import Any, OrderedDict

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, nn

from anomalib.utils.metrics import get_normalization_metrics, get_thresholding_metrics

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
        """Step function called during :meth:`~pytorch_lightning.trainer.trainer.Trainer.predict`.

        By default, it calls :meth:`~pytorch_lightning.core.lightning.LightningModule.forward`.
        Override to add any processing logic.

        Args:
            batch (Any): Current batch
            batch_idx (int): Index of current batch
            dataloader_idx (int): Index of the current dataloader

        Return:
            Predicted output
        """
        del batch_idx, dataloader_idx  # These variables are not used.

        outputs: Tensor | dict[str, Any] = self.validation_step(batch)
        return outputs

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

    def _load_extra_keys(self, state_dict: OrderedDict[str, Tensor]) -> None:
        """Adds thresholds and normalization metrics if they are in the state dict.

        Args:
            state_dict (OrderedDict[str, Tensor]): State dict of the model.
        """
        for key in state_dict.keys():
            if key.startswith("normalization") and not hasattr(self, "normalization_metrics"):
                self.normalization_metrics = get_normalization_metrics(key)
            elif key.startswith("image_threshold") and not hasattr(self, "image_threshold"):
                self.image_threshold = get_thresholding_metrics()
            elif key.startswith("pixel_threshold") and not hasattr(self, "pixel_threshold"):
                self.pixel_threshold = get_thresholding_metrics()

    def load_state_dict(self, state_dict: OrderedDict[str, Tensor], strict: bool = True):
        """Load state dict from checkpoint.
        Ensures that normalization and thresholding attributes is properly setup before model is loaded.
        """
        # Used to load missing normalization and threshold parameters
        self._load_extra_keys(state_dict)
        return super().load_state_dict(state_dict, strict=strict)
