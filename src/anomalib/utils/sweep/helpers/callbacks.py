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
from anomalib.utils.metrics import AnomalyScoreMeanVarThreshold


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
        # This is for AnomalyScoreMeanVarThreshold
        std_dev_ratio = (
            config.metrics.threshold.std_dev_ratio
            if "std_dev_ratio" in config.metrics.threshold.keys()
            else AnomalyScoreMeanVarThreshold.DEFAULT_POSITIVE_RATE
        )
        meanvar_mode = (
            config.metrics.threshold.meanvar_mode
            if "meanvar_mode" in config.metrics.threshold.keys()
            else AnomalyScoreMeanVarThreshold.MEANVAR_MODE
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
        # This is for AnomalyScoreMeanVarThreshold
        std_dev_ratio = (
            config.post_processing.std_dev_ratio
            if "std_dev_ratio" in config.post_processing.keys()
            else AnomalyScoreMeanVarThreshold.DEFAULT_POSITIVE_RATE
        )
        meanvar_mode = (
            config.post_processing.meanvar_mode
            if "meanvar_mode" in config.post_processing.keys()
            else AnomalyScoreMeanVarThreshold.MEANVAR_MODE
        )

    post_processing_configuration_callback = PostProcessingConfigurationCallback(
        normalization_method=normalization_method,
        manual_image_threshold=image_threshold,
        manual_pixel_threshold=pixel_threshold,
        std_dev_ratio=std_dev_ratio,
        meanvar_mode=meanvar_mode,
    )

    callbacks.append(post_processing_configuration_callback)

    metrics_configuration_callback = MetricsConfigurationCallback(
        task=config.dataset.task, image_metrics=image_metrics, pixel_metrics=pixel_metrics
    )
    callbacks.append(metrics_configuration_callback)

    return callbacks
