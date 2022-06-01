"""Metrics Configuration Callback."""

# Copyright (C) 2022 Intel Corporation
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


from typing import List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from anomalib.models.components.base.anomaly_module import AnomalyModule
from anomalib.utils.metrics import metric_collection_from_names

__all__ = ["MetricsConfigurationCallback"]


class MetricsConfigurationCallback(Callback):
    """Metrics Configuration Callback."""

    def __init__(
        self,
        adaptive_threshold: bool,
        default_image_threshold: Optional[float] = None,
        default_pixel_threshold: Optional[float] = None,
        image_metric_names: Optional[List[str]] = None,
        pixel_metric_names: Optional[List[str]] = None,
    ):
        """Create image and pixel-level AnomalibMetricsCollection.

        This callback creates AnomalibMetricsCollection based on the
            list of strings provided for image and pixel-level metrics.
        After these MetricCollections are created, the callback assigns
        these to the lightning module.

        Args:
            adaptive_threshold (bool): Flag indicating whether threshold should be adaptive.
            default_image_threshold (Optional[float]): Default image threshold value.
            default_pixel_threshold (Optional[float]): Default pixel threshold value.
            image_metric_names (Optional[List[str]]): List of image-level metrics.
            pixel_metric_names (Optional[List[str]]): List of pixel-level metrics.
        """

        self.image_metric_names = image_metric_names
        self.pixel_metric_names = pixel_metric_names

        assert (
            adaptive_threshold or default_image_threshold and default_pixel_threshold
        ), "Default thresholds must be specified when adaptive threshold is disabled."
        self.adaptive_threshold = adaptive_threshold
        self.default_image_threshold = default_image_threshold
        self.default_pixel_threshold = default_pixel_threshold

    def setup(
        self,
        _trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: Optional[str] = None,  # pylint: disable=unused-argument
    ) -> None:
        """Setup image and pixel-level AnomalibMetricsCollection within Anomalib Model.

        Args:
            _trainer (pl.Trainer): PyTorch Lightning Trainer
            pl_module (pl.LightningModule): Anomalib Model that inherits pl LightningModule.
            stage (Optional[str], optional): fit, validate, test or predict. Defaults to None.
        """
        image_metric_names = [] if self.image_metric_names is None else self.image_metric_names
        pixel_metric_names = [] if self.pixel_metric_names is None else self.pixel_metric_names

        if isinstance(pl_module, AnomalyModule):
            pl_module.adaptive_threshold = self.adaptive_threshold
            if not self.adaptive_threshold:
                # pylint: disable=not-callable
                pl_module.image_threshold.value = torch.tensor(self.default_image_threshold).cpu()
                pl_module.pixel_threshold.value = torch.tensor(self.default_pixel_threshold).cpu()

            pl_module.image_metrics = metric_collection_from_names(image_metric_names, "image_")
            pl_module.pixel_metrics = metric_collection_from_names(pixel_metric_names, "pixel_")

            pl_module.image_metrics.set_threshold(pl_module.image_threshold.value)
            pl_module.pixel_metrics.set_threshold(pl_module.pixel_threshold.value)
