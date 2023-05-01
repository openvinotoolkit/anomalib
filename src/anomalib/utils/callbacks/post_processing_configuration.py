"""Post-Processing Configuration Callback."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import logging
from importlib import import_module

from pytorch_lightning import Callback, LightningModule, Trainer

from anomalib.models.components.base.anomaly_module import AnomalyModule
from anomalib.utils.metrics.thresholding import AdaptiveScoreThreshold, BaseAnomalyScoreThreshold

logger = logging.getLogger(__name__)

__all__ = ["PostProcessingConfigurationCallback"]


class PostProcessingConfigurationCallback(Callback):
    """Post-Processing Configuration Callback.

    Args:
        image_threshold_class (str | None): Threshold class. Defaults to Adaptive Thresholding.
        image_threshold_args (dict | None): Arguments for the thresholding class.
        pixel_threshold_class (str): Threshold class. Defaults to None as not all models use pixel thresholding.
        pixel_threshold_args (dict | None): Arguments for the thresholding class.
    """

    def __init__(
        self,
        image_threshold_class: str | None = "AdaptiveScoreThreshold",
        image_threshold_args: dict | None = None,
        pixel_threshold_class: str | None = None,
        pixel_threshold_args: dict | None = None,
    ) -> None:
        super().__init__()

        self.image_threshold = self._get_threshold_method(image_threshold_class, image_threshold_args)
        self.pixel_threshold = self._get_threshold_method(pixel_threshold_class, pixel_threshold_args)

    def _get_threshold_method(
        self, threshold_class: str | None, threshold_args: dict | None = None
    ) -> BaseAnomalyScoreThreshold:
        """Gets the thresholding class based on the class_path

        Args:
            threshold_class (str | None): Threshold class. If None, Adaptive Thresholding is used.
            threshold_args (dict | None): Arguments for the thresholding class. Defaults to None.

        Returns:
            BaseAnomalyScoreThreshold: Thresholding class
        """
        threshold_method: BaseAnomalyScoreThreshold
        if threshold_class is None:
            threshold_method = AdaptiveScoreThreshold()
        else:
            try:
                if len(threshold_class.split(".")) > 1:  # When the entire class path is provided
                    threshold_module = import_module(".".join(threshold_class.split(".")[:-1]))
                    _threshold_class = getattr(threshold_module, threshold_class.split(".")[-1])
                else:
                    threshold_module = import_module("anomalib.utils.metrics.thresholding")
                    _threshold_class = getattr(threshold_module, threshold_class)
            except (AttributeError, ModuleNotFoundError) as exception:
                raise Exception(f"Threshold class {threshold_class} not found") from exception

            if threshold_args is None:
                threshold_args = {}
            threshold_method = _threshold_class(**threshold_args)

        return threshold_method

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str | None = None) -> None:
        """Setup post-processing configuration within Anomalib Model.

        Args:
            trainer (Trainer): PyTorch Lightning Trainer
            pl_module (LightningModule): Anomalib Model that inherits pl LightningModule.
            stage (str | None, optional): fit, validate, test or predict. Defaults to None.
        """
        del trainer, stage  # These variables are not used.

        if isinstance(pl_module, AnomalyModule):
            if not hasattr(pl_module, "image_threshold"):
                pl_module.image_threshold = self.image_threshold
            if not hasattr(pl_module, "pixel_threshold"):
                pl_module.pixel_threshold = self.pixel_threshold
