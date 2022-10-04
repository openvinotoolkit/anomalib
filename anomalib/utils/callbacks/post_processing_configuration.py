"""Post-Processing Configuration Callback."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging
from typing import Optional

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY

from anomalib.models.components.base.anomaly_module import AnomalyModule

logger = logging.getLogger(__name__)

__all__ = ["PostProcessingConfigurationCallback"]


@CALLBACK_REGISTRY
class PostProcessingConfigurationCallback(Callback):
    """Post-Processing Configuration Callback.

    Args:
        normalization_method(Optional[str]): Normalization method. <None, min_max, cdf>
        adaptive_threshold (bool): Flag indicating whether threshold should be adaptive.
        default_image_threshold (Optional[float]): Default image threshold value.
        default_pixel_threshold (Optional[float]): Default pixel threshold value.
    """

    def __init__(
        self,
        normalization_method: str = "min_max",
        adaptive_threshold: bool = True,
        default_image_threshold: Optional[float] = None,
        default_pixel_threshold: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.normalization_method = normalization_method

        assert (
            adaptive_threshold or default_image_threshold is not None and default_pixel_threshold is not None
        ), "Default thresholds must be specified when adaptive threshold is disabled."

        self.adaptive_threshold = adaptive_threshold
        self.default_image_threshold = default_image_threshold
        self.default_pixel_threshold = default_pixel_threshold

    # pylint: disable=unused-argument
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        """Setup post-processing configuration within Anomalib Model.

        Args:
            trainer (Trainer): PyTorch Lightning Trainer
            pl_module (LightningModule): Anomalib Model that inherits pl LightningModule.
            stage (Optional[str], optional): fit, validate, test or predict. Defaults to None.
        """
        if isinstance(pl_module, AnomalyModule):
            pl_module.adaptive_threshold = self.adaptive_threshold
            if pl_module.adaptive_threshold is False:
                pl_module.image_threshold.value = torch.tensor(self.default_image_threshold).cpu()
                pl_module.pixel_threshold.value = torch.tensor(self.default_pixel_threshold).cpu()
