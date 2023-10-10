"""Implements custom trainer for Anomalib."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from lightning import Callback
from lightning.pytorch import LightningDataModule, Trainer
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


class UnassignedError(Exception):
    ...


class _TrainerArgumentsCache:
    """Cache arguments.

    Since the Engine class accepts PyTorch Lightning Trainer arguments, we store these arguments using this class
    before the trainer is instantiated.

    Args:
        (**kwargs): Trainer arguments that are cached

    Example:
        >>> conf = OmegaConf.load("config.yaml")
        >>> cache =  _TrainerArgumentsCache(**conf.trainer)
        >>> cache.args
        {
            ...
            'max_epochs': 100,
            'val_check_interval': 0
        }
        >>> model = Padim(layers=["layer1", "layer2", "layer3"], input_size=(256, 256), backbone="resnet18")
        >>> cache.update(model)
        Overriding max_epochs from 100 with 1 for Padim
        Overriding val_check_interval from 0 with 1.0 for Padim
        >>> cache.args
        {
            ...
            'max_epochs': 1,
            'val_check_interval': 1.0
        }
    """

    def __init__(self, **kwargs):
        self._cached_args = {**kwargs}

    def update(self, model: AnomalyModule):
        """Replace cached arguments with arguments retrieved from the model.

        Args:
            model (AnomalyModule): The model used for training
        """
        for key, value in model.trainer_arguments.items():
            if key in self._cached_args:
                if self._cached_args[key] != value:
                    log.info(
                        f"Overriding {key} from {self._cached_args[key]} with {value} for {model.__class__.__name__}"
                    )
                self._cached_args[key] = value

    def requires_update(self, model: AnomalyModule) -> bool:
        for key, value in model.trainer_arguments.items():
            if key in self._cached_args and self._cached_args[key] != value:
                return True
        return False

    @property
    def args(self) -> dict[str, Any]:
        return self._cached_args


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
        normalization: NormalizationMethod | DictConfig | Callback | str = NormalizationMethod.MIN_MAX,
        threshold: BaseThreshold
        | tuple[BaseThreshold, BaseThreshold]
        | DictConfig
        | ListConfig
        | str = F1AdaptiveThreshold(),
        task: TaskType = TaskType.SEGMENTATION,
        image_metrics: str | list[str] | None = None,
        pixel_metrics: str | list[str] | None = None,
        visualization: DictConfig | None = None,
        **kwargs,
    ) -> None:
        self._cache = _TrainerArgumentsCache(callbacks=[*callbacks], **kwargs)
        self.normalization = normalization
        self.threshold = threshold
        self.task = task
        self.image_metric_names = image_metrics
        self.pixel_metric_names = pixel_metrics
        self.visualization = visualization

        self._trainer: Trainer | None = None

    @property
    def trainer(self):
        if not self._trainer:
            msg = "``self.trainer`` is not assigned yet."
            raise UnassignedError(msg)
        return self._trainer

    def _setup_trainer(self, model: AnomalyModule):
        """Instantiates the trainer based on the model parameters."""
        if self._cache.requires_update(model) or self._trainer is None:
            self._cache.update(model)
            self._trainer = Trainer(**self._cache.args)
            # Callbacks need to be setup later as they depend on default_root_dir from the trainer
            self._setup_anomalib_callbacks()

    def _setup_anomalib_callbacks(self):
        """Setup callbacks for the trainer."""
        _callbacks: list[Callback] = [_PostProcessorCallback()]
        normalization_callback = get_normalization_callback(self.normalization)
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
        train_dataloaders: TRAIN_DATALOADERS | LightningDataModule | None = None,
        val_dataloaders: EVAL_DATALOADERS | None = None,
        datamodule: LightningDataModule | None = None,
        ckpt_path: str | None = None,
    ) -> None:
        self._setup_trainer(model)
        self.trainer.fit(model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)

    def validate(
        self,
        model: AnomalyModule | None = None,
        dataloaders: EVAL_DATALOADERS | LightningDataModule | None = None,
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
        dataloaders: EVAL_DATALOADERS | LightningDataModule | None = None,
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
        dataloaders: EVAL_DATALOADERS | LightningDataModule | None = None,
        datamodule: LightningDataModule | None = None,
        return_predictions: bool | None = None,
        ckpt_path: str | None = None,
    ) -> _PREDICT_OUTPUT | None:
        if model:
            self._setup_trainer(model)
        return self.trainer.predict(model, dataloaders, datamodule, return_predictions, ckpt_path)
