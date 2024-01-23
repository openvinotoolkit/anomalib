"""Get callbacks related to sweep."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from omegaconf import DictConfig, ListConfig
from pytorch_lightning import Callback

from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    PostProcessingConfigurationCallback,
)
from anomalib.utils.callbacks.timer import TimerCallback
from anomalib.utils.metrics import AnomalyScoreGaussianMixtureThreshold


def get_sweep_callbacks(config: DictConfig | ListConfig) -> list[Callback]:
    """Gets callbacks relevant to sweep.

    Args:
        config (DictConfig | ListConfig): Model config loaded from anomalib

    Returns:
        list[Callback]: List of callbacks
    """
    callbacks: list[Callback] = [TimerCallback()]
    # Add metric configuration to the model via MetricsConfigurationCallback

    # TODO: Remove this once the old CLI is deprecated.
    if isinstance(config, DictConfig):
        image_metrics = config.metrics.image if "image" in config.metrics.keys() else None
        pixel_metrics = config.metrics.pixel if "pixel" in config.metrics.keys() else None
        image_threshold = (
            config.metrics.threshold.manual_image if "manual_image" in config.metrics.threshold.keys() else None
        )
        pixel_threshold = (
            config.metrics.threshold.manual_pixel if "manual_pixel" in config.metrics.threshold.keys() else None
        )
        normalization_method = config.model.normalization_method

        # For Gaussian Mixture Estimation of threshold.
        image_anomalous_rate = (
            config.metrics.threshold.image_anomalous_rate
            if "image_anomalous_rate" in config.metrics.threshold.keys()
            else AnomalyScoreGaussianMixtureThreshold.DEFAULT_ANOMALOUS_RATE
        )
        pixel_anomalous_rate = (
            config.metrics.threshold.pixel_anomalous_rate
            if "pixel_anomalous_rate" in config.metrics.threshold.keys()
            else AnomalyScoreGaussianMixtureThreshold.DEFAULT_ANOMALOUS_RATE
        )
        image_n_components = (
            config.metrics.threshold.image_n_components
            if "image_n_components" in config.metrics.threshold.keys()
            else AnomalyScoreGaussianMixtureThreshold.DEFAULT_N_COMPONENTS
        )
        pixel_n_components = (
            config.metrics.threshold.pixel_n_components
            if "pixel_n_components" in config.metrics.threshold.keys()
            else AnomalyScoreGaussianMixtureThreshold.DEFAULT_N_COMPONENTS
        )

    # NOTE: This is for the new anomalib CLI.
    else:
        image_metrics = config.metrics.image_metrics if "image_metrics" in config.metrics else None
        pixel_metrics = config.metrics.pixel_metrics if "pixel_metrics" in config.metrics else None
        image_threshold = (
            config.post_processing.manual_image_threshold if "image_default" in config.post_processing.keys() else None
        )
        pixel_threshold = (
            config.post_processing.manual_pixel_threshold if "pixel_default" in config.post_processing.keys() else None
        )
        normalization_method = config.post_processing.normalization_method

        # For Gaussian Mixture Estimation of threshold.
        image_anomalous_rate = (
            config.post_processing.image_anomalous_rate
            if "image_anomalous_rate" in config.metrics.threshold.keys()
            else AnomalyScoreGaussianMixtureThreshold.DEFAULT_ANOMALOUS_RATE
        )
        pixel_anomalous_rate = (
            config.post_processing.pixel_anomalous_rate
            if "pixel_anomalous_rate" in config.metrics.threshold.keys()
            else AnomalyScoreGaussianMixtureThreshold.DEFAULT_ANOMALOUS_RATE
        )
        image_n_components = (
            config.post_processing.image_n_components
            if "image_n_components" in config.metrics.threshold.keys()
            else AnomalyScoreGaussianMixtureThreshold.DEFAULT_N_COMPONENTS
        )
        pixel_n_components = (
            config.post_processing.pixel_n_components
            if "pixel_n_components" in config.metrics.threshold.keys()
            else AnomalyScoreGaussianMixtureThreshold.DEFAULT_N_COMPONENTS
        )

    post_processing_configuration_callback = PostProcessingConfigurationCallback(
        normalization_method=normalization_method,
        manual_image_threshold=image_threshold,
        manual_pixel_threshold=pixel_threshold,
        image_anomalous_rate=image_anomalous_rate,
        pixel_anomalous_rate=pixel_anomalous_rate,
        image_n_components=image_n_components,
        pixel_n_components=pixel_n_components,
    )
    callbacks.append(post_processing_configuration_callback)

    metrics_configuration_callback = MetricsConfigurationCallback(
        task=config.dataset.task, image_metrics=image_metrics, pixel_metrics=pixel_metrics
    )
    callbacks.append(metrics_configuration_callback)

    return callbacks
