"""Get callbacks related to sweep."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from typing import List, Union

from omegaconf import DictConfig, ListConfig
from pytorch_lightning import Callback

from anomalib.utils.callbacks import MetricsConfigurationCallback
from anomalib.utils.callbacks.timer import TimerCallback


def get_sweep_callbacks(config: Union[ListConfig, DictConfig]) -> List[Callback]:
    """Gets callbacks relevant to sweep.

    Args:
        config (Union[DictConfig, ListConfig]): Model config loaded from anomalib

    Returns:
        List[Callback]: List of callbacks
    """
    callbacks: List[Callback] = [TimerCallback()]
    # Add metric configuration to the model via MetricsConfigurationCallback
    image_metric_names = config.metrics.image if "image" in config.metrics.keys() else None
    pixel_metric_names = config.metrics.pixel if "pixel" in config.metrics.keys() else None
    image_threshold = (
        config.metrics.threshold.image_default if "image_default" in config.metrics.threshold.keys() else None
    )
    pixel_threshold = (
        config.metrics.threshold.pixel_default if "pixel_default" in config.metrics.threshold.keys() else None
    )
    metrics_callback = MetricsConfigurationCallback(
        config.metrics.threshold.adaptive,
        image_threshold,
        pixel_threshold,
        image_metric_names,
        pixel_metric_names,
    )
    callbacks.append(metrics_callback)

    return callbacks
