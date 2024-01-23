"""Post-Processing Configuration Callback."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import logging

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

from anomalib.models.components.base.anomaly_module import AnomalyModule
from anomalib.post_processing import NormalizationMethod, ThresholdMethod

logger = logging.getLogger(__name__)

__all__ = ["PostProcessingConfigurationCallback"]


class PostProcessingConfigurationCallback(Callback):
    """Post-Processing Configuration Callback.

    Args:
        normalization_method(NormalizationMethod): Normalization method. <none, min_max, cdf>
        threshold_method (ThresholdMethod): Flag indicating whether threshold should be manual or adaptive.
        manual_image_threshold (float | None): Default manual image threshold value.
        manual_pixel_threshold (float | None): Default manual pixel threshold value.
    """

    def __init__(
        self,
        normalization_method: NormalizationMethod = NormalizationMethod.MIN_MAX,
        threshold_method: ThresholdMethod = ThresholdMethod.ADAPTIVE,
        manual_image_threshold: float | None = None,
        manual_pixel_threshold: float | None = None,
        std_dev_ratio: float | None = None,
        meanvar_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.normalization_method = normalization_method

        if threshold_method == ThresholdMethod.ADAPTIVE and all(
            i is not None for i in (manual_image_threshold, manual_pixel_threshold)
        ):
            raise ValueError(
                "When `threshold_method` is set to `adaptive`, `manual_image_threshold` and `manual_pixel_threshold` "
                "must not be set."
            )

        if threshold_method == ThresholdMethod.MANUAL and all(
            i is None for i in (manual_image_threshold, manual_pixel_threshold)
        ):
            raise ValueError(
                "When `threshold.method` is set to `manual`, `manual_image_threshold` and `manual_pixel_threshold` "
                "must be set."
            )

        if threshold_method == ThresholdMethod.MEANVAR and any(
            i is not None for i in (manual_image_threshold, manual_pixel_threshold)
        ):
            raise ValueError(
                "When ``threshold.method` is set to `meanvar`, `manual_image_threshold` or `manual_pixel_threshold` "
                "must not be set."
            )

        self.threshold_method = threshold_method
        self.manual_image_threshold = manual_image_threshold
        self.manual_pixel_threshold = manual_pixel_threshold
        self.std_dev_ratio = std_dev_ratio
        self.meanvar_mode = meanvar_mode

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str | None = None) -> None:
        """Setup post-processing configuration within Anomalib Model.

        Args:
            trainer (Trainer): PyTorch Lightning Trainer
            pl_module (LightningModule): Anomalib Model that inherits pl LightningModule.
            stage (str | None, optional): fit, validate, test or predict. Defaults to None.
        """
        del trainer, stage  # These variables are not used.

        if isinstance(pl_module, AnomalyModule):
            pl_module.threshold_method = self.threshold_method
            if pl_module.threshold_method == ThresholdMethod.MANUAL:
                pl_module.image_threshold.value = torch.tensor(self.manual_image_threshold).cpu()
                pl_module.pixel_threshold.value = torch.tensor(self.manual_pixel_threshold).cpu()
            if pl_module.threshold_method == ThresholdMethod.MEANVAR:
                image_threshold = pl_module.image_threshold
                pixel_threshold = pl_module.pixel_threshold
                image_threshold.std_dev_ratio = self.std_dev_ratio
                image_threshold.meanvar_mode = self.meanvar_mode
                image_threshold.std_dev_ratio = self.std_dev_ratio
                image_threshold.meanvar_mode = self.meanvar_mode
                pixel_threshold.std_dev_ratio = self.std_dev_ratio
                pixel_threshold.meanvar_mode = self.meanvar_mode
                pixel_threshold.std_dev_ratio = self.std_dev_ratio
                pixel_threshold.meanvar_mode = self.meanvar_mode
