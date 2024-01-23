"""Post-Processing Configuration Callback."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import logging

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

from anomalib.models.components.base.anomaly_module import AnomalyModule
from anomalib.post_processing import ADAPTIVE_THRESHOLD_METHOD_MAP, NormalizationMethod, ThresholdMethod
from anomalib.utils.metrics import AnomalyScoreGaussianMixtureThreshold

logger = logging.getLogger(__name__)

__all__ = ["PostProcessingConfigurationCallback"]


class PostProcessingConfigurationCallback(Callback):
    """Post-Processing Configuration Callback.

    Args:
        normalization_method(NormalizationMethod): Normalization method. <none, min_max, cdf>
        threshold_method (ThresholdMethod): Flag indicating whether threshold should be manual or adaptive.
        manual_image_threshold (float | None): Default manual image threshold value.
        manual_pixel_threshold (float | None): Default manual pixel threshold value.
        image_anomalous_rate (float | None): Anticipated image anomalous rate for Gaussian Mixture based threshold.
        pixel_anomalous_rate (float | None): Anticipated pixel anomalous rate for Gaussian Mixture based threshold.
        image_n_components (int): Number of mixture components of Gaussian Mixture based threshold for images.
        pixel_n_components (int): Number of mixture components of Gaussian Mixture based threshold for pixels.
    """

    def __init__(
        self,
        normalization_method: NormalizationMethod = NormalizationMethod.MIN_MAX,
        threshold_method: ThresholdMethod = ThresholdMethod.ADAPTIVE,
        manual_image_threshold: float | None = None,
        manual_pixel_threshold: float | None = None,
        image_anomalous_rate: float | None = AnomalyScoreGaussianMixtureThreshold.DEFAULT_ANOMALOUS_RATE,
        pixel_anomalous_rate: float | None = AnomalyScoreGaussianMixtureThreshold.DEFAULT_ANOMALOUS_RATE,
        image_n_components: int = AnomalyScoreGaussianMixtureThreshold.DEFAULT_N_COMPONENTS,
        pixel_n_components: int = AnomalyScoreGaussianMixtureThreshold.DEFAULT_N_COMPONENTS,
    ) -> None:
        super().__init__()
        self.normalization_method = normalization_method

        if threshold_method in ADAPTIVE_THRESHOLD_METHOD_MAP and all(
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
                "When `threshold_method` is set to `manual`, `manual_image_threshold` and `manual_pixel_threshold` "
                "must be set."
            )

        self.threshold_method = threshold_method
        self.manual_image_threshold = manual_image_threshold
        self.manual_pixel_threshold = manual_pixel_threshold
        self.image_anomalous_rate = image_anomalous_rate
        self.pixel_anomalous_rate = pixel_anomalous_rate
        self.image_n_components = image_n_components
        self.pixel_n_components = pixel_n_components

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
            if pl_module.threshold_method == ThresholdMethod.GAUSSIAN_MIXTURE:
                image_threshold: AnomalyScoreGaussianMixtureThreshold = pl_module.image_threshold
                pixel_threshold: AnomalyScoreGaussianMixtureThreshold = pl_module.pixel_threshold
                image_threshold.anomalous_rate = self.image_anomalous_rate
                pixel_threshold.anomalous_rate = self.pixel_anomalous_rate
                image_threshold.n_components = self.image_n_components
                pixel_threshold.n_components = self.pixel_n_components
