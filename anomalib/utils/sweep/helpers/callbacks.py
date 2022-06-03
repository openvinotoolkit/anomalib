"""Get callbacks related to sweep."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.


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
