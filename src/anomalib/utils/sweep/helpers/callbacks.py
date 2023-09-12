"""Get callbacks related to sweep."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from lightning.pytorch import Callback
from omegaconf import DictConfig, ListConfig

from anomalib.utils.callbacks import MetricsConfigurationCallback
from anomalib.utils.callbacks.timer import TimerCallback


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

    # NOTE: This is for the new anomalib CLI.
    else:
        image_metrics = config.metrics.image_metrics if "image_metrics" in config.metrics else None
        pixel_metrics = config.metrics.pixel_metrics if "pixel_metrics" in config.metrics else None

    metrics_configuration_callback = MetricsConfigurationCallback(
        task=config.dataset.task, image_metrics=image_metrics, pixel_metrics=pixel_metrics
    )
    callbacks.append(metrics_configuration_callback)

    return callbacks
