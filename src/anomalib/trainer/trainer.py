"""Implements custom trainer for Anomalib."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from lightning import Callback
from lightning.pytorch import Trainer
from lightning.pytorch.trainer.connectors.callback_connector import _CallbackConnector
from omegaconf import DictConfig, ListConfig

from anomalib.data import TaskType
from anomalib.models import AnomalyModule
from anomalib.post_processing import NormalizationMethod
from anomalib.utils.callbacks import get_visualization_callbacks
from anomalib.utils.callbacks.metrics import _MetricsCallback
from anomalib.utils.callbacks.normalization import get_normalization_callback
from anomalib.utils.callbacks.post_processor import _PostProcessorCallback
from anomalib.utils.callbacks.thresholding import _ThresholdCallback
from anomalib.utils.metrics.threshold import BaseThreshold, F1AdaptiveThreshold

log = logging.getLogger(__name__)


class AnomalibTrainer(Trainer):
    """Anomalib trainer.

    Note:
        Refer to PyTorch Lightning's Trainer for a list of parameters for details on other Trainer parameters.

    Args:
        callbacks: Add a callback or list of callbacks.
    """

    def __init__(
        self,
        callbacks: list[Callback] = [],
        normalizer: NormalizationMethod | DictConfig | Callback | str = NormalizationMethod.MIN_MAX,
        threshold: BaseThreshold
        | tuple[BaseThreshold, BaseThreshold]
        | DictConfig
        | ListConfig
        | str = F1AdaptiveThreshold(),
        task: TaskType = TaskType.SEGMENTATION,
        image_metrics: list[str] | None = None,
        pixel_metrics: list[str] | None = None,
        visualization: DictConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(callbacks=callbacks, **kwargs)
        self.normalizer = normalizer
        self.threshold = threshold
        self.task = task
        self.image_metric_names = image_metrics
        self.pixel_metric_names = pixel_metrics
        self.visualization = visualization

        self.lightning_module: AnomalyModule

        self._setup_callbacks()

    def _setup_callbacks(self):
        """Setup callbacks for the trainer."""

        _callbacks: list[Callback] = [_PostProcessorCallback()]
        normalization_callback = get_normalization_callback(self.normalizer)
        if normalization_callback is not None:
            _callbacks.append(normalization_callback)

        _callbacks.append(_ThresholdCallback(self.threshold))
        _callbacks.append(_MetricsCallback(self.task, self.image_metric_names, self.pixel_metric_names))

        if self.visualization is not None:
            image_save_path = self.visualization.pop("image_save_path", None)
            if image_save_path is None:
                image_save_path = self.default_root_dir + "/images"
            _callbacks += get_visualization_callbacks(
                task=self.task, image_save_path=image_save_path, **self.visualization
            )

        self.callbacks = _CallbackConnector._reorder_callbacks(self.callbacks + _callbacks)
