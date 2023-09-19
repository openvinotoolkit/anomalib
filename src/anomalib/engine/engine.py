"""Implements custom trainer for Anomalib."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from lightning import Callback
from lightning.pytorch import LightningDataModule, Trainer
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.trainer.connectors.callback_connector import _CallbackConnector
from lightning.pytorch.utilities.types import _EVALUATE_OUTPUT, _PREDICT_OUTPUT, EVAL_DATALOADERS, TRAIN_DATALOADERS
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


class UnassignedException(Exception):
    ...


class _Cache:
    """Cache arguments.

    Args:
        (**kwargs): Trainer arguments that are cached

    Example:
        >>> cache = _Cache(max_epochs=100, devices=0)
        >>> model = Padim()
        >>> cache.update(model)
        >>> Trainer(**cache.params)
    """

    def __init__(self, **kwargs):
        self._cache = {**kwargs}

    def update(self, model: AnomalyModule):
        """Replace cached arguments with arguments retrieved from the model.

        Args:
            model (AnomalyModule): The model used for training
        """
        for key, value in model.trainer_arguments.items():
            if key in self._cache:
                if self._cache[key] != value:
                    log.info(f"Overriding {key} from {self._cache[key]} with {value} for {model.__class__.__name__}")
                self._cache[key] = value

    def needs_override(self, model: AnomalyModule) -> bool:
        for key, value in model.trainer_arguments.items():
            if key in self._cache and self._cache[key] != value:
                return True
        return False

    @property
    def params(self) -> dict[str, Any]:
        return self._cache


class Engine:
    """Anomalib Engine.

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
        self._cache = _Cache(callbacks=[*callbacks, RichProgressBar()], **kwargs)
        self.normalizer = normalizer
        self.threshold = threshold
        self.task = task
        self.image_metric_names = image_metrics
        self.pixel_metric_names = pixel_metrics
        self.visualization = visualization

        self.lightning_module: AnomalyModule
        self._trainer: Trainer | None = None

    @property
    def trainer(self):
        if not self._trainer:
            msg = "``self.trainer`` is not assigned yet."
            raise UnassignedException(msg)
        return self._trainer

    def _setup_trainer(self, model: AnomalyModule):
        """Instantiates the trainer based on the model parameters."""
        if self._cache.needs_override(model) or self._trainer is None:
            self._cache.update(model)
            self._trainer = Trainer(**self._cache.params)
            # Callbacks need to be setup later as they depend on default_root_dir from the trainer
            self._setup_anomalib_callbacks()

    def _setup_anomalib_callbacks(self):
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
                image_save_path = self.trainer.default_root_dir + "/images"
            _callbacks += get_visualization_callbacks(
                task=self.task, image_save_path=image_save_path, **self.visualization
            )

        self.trainer.callbacks = _CallbackConnector._reorder_callbacks(self.trainer.callbacks + _callbacks)

    def fit(
        self,
        model: AnomalyModule,
        train_dataloaders: TRAIN_DATALOADERS | LightningDataModule = None,
        val_dataloaders: EVAL_DATALOADERS = None,
        datamodule: LightningDataModule | None = None,
        ckpt_path: str | None = None,
    ) -> None:
        self._setup_trainer(model)
        self.trainer.fit(model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)

    def validate(
        self,
        model: AnomalyModule | None = None,
        dataloaders: EVAL_DATALOADERS | LightningDataModule = None,
        ckpt_path: str | None = None,
        verbose: bool = True,
        datamodule: LightningDataModule | None = None,
    ) -> _EVALUATE_OUTPUT | None:
        if model:
            self._setup_trainer(model)
        return self.trainer.validate(model, dataloaders, ckpt_path, verbose, datamodule)

    def test(
        self,
        model: AnomalyModule | None = None,
        dataloaders: EVAL_DATALOADERS | LightningDataModule = None,
        ckpt_path: str | None = None,
        verbose: bool = True,
        datamodule: LightningDataModule | None = None,
    ) -> _EVALUATE_OUTPUT:
        if model:
            self._setup_trainer(model)
        return self.trainer.test(model, dataloaders, ckpt_path, verbose, datamodule)

    def predict(
        self,
        model: AnomalyModule | None = None,
        dataloaders: EVAL_DATALOADERS | LightningDataModule = None,
        datamodule: LightningDataModule | None = None,
        return_predictions: bool | None = None,
        ckpt_path: str | None = None,
    ) -> _PREDICT_OUTPUT | None:
        if model:
            self._setup_trainer(model)
        return self.trainer.predict(model, dataloaders, datamodule, return_predictions, ckpt_path)
