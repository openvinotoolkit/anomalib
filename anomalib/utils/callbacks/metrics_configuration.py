"""Metrics Configuration Callback."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from typing import List, Optional, Union

import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY

from anomalib.models import AnomalyModule
from anomalib.utils.metrics import metric_collection_from_names

__all__ = ["MetricsConfigurationCallback"]


@CALLBACK_REGISTRY
class MetricsConfigurationCallback(Callback):
    """Metrics Configuration Callback."""

    def __init__(
        self,
        thresholding: Union[str, DictConfig],
        image_metric_names: Optional[List[str]] = None,
        pixel_metric_names: Optional[List[str]] = None,
        normalization_method: str = "min_max",
    ):
        """Create image and pixel-level AnomalibMetricsCollection.

        This callback creates AnomalibMetricsCollection based on the
            list of strings provided for image and pixel-level metrics.
        After these MetricCollections are created, the callback assigns
        these to the lightning module.

        Args:
            thresholding (Union[str, DictConfig]): Parameters related to thresholding.
            image_metric_names (Optional[List[str]]): List of image-level metrics.
            pixel_metric_names (Optional[List[str]]): List of pixel-level metrics.
            normalization_method(Optional[str]): Normalization method. <None, min_max, cdf>
        """
        # TODO: https://github.com/openvinotoolkit/anomalib/issues/384
        self.image_metric_names = image_metric_names
        self.pixel_metric_names = pixel_metric_names

        # TODO: https://github.com/openvinotoolkit/anomalib/issues/384
        # TODO: This is a workaround. normalization-method is actually not used in metrics.
        #   It's only accessed from `before_instantiate` method in `AnomalibCLI` to configure
        #   its callback.
        self.normalization_method = normalization_method

        self.thresholding = thresholding

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
            pl_module.threshold = self.thresholding

            pl_module.image_metrics = metric_collection_from_names(image_metric_names, "image_")
            pl_module.pixel_metrics = metric_collection_from_names(pixel_metric_names, "pixel_")

            pl_module.image_metrics.set_threshold(pl_module.image_threshold.value)
            pl_module.pixel_metrics.set_threshold(pl_module.pixel_threshold.value)
