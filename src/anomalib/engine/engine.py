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
from anomalib.utils.metrics.threshold import BaseThreshold

log = logging.getLogger(__name__)


class UnassignedError(Exception):
    """Unassigned error."""

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

    def __init__(self, **kwargs) -> None:
        self._cached_args = {**kwargs}

    def update(self, model: AnomalyModule) -> None:
        """Replace cached arguments with arguments retrieved from the model.

        Args:
            model (AnomalyModule): The model used for training
        """
        for key, value in model.trainer_arguments.items():
            if key in self._cached_args and self._cached_args[key] != value:
                log.info(
                    f"Overriding {key} from {self._cached_args[key]} with {value} for {model.__class__.__name__}",
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
        callbacks (list[Callback]): Add a callback or list of callbacks.
        normalization (NormalizationMethod | DictConfig | Callback | str, optional): Normalization method.
            Defaults to NormalizationMethod.MIN_MAX.
        threshold (BaseThreshold | tuple[BaseThreshold, BaseThreshold] | DictConfig | ListConfig | str, optional):
            Thresholding method. Defaults to "F1AdaptiveThreshold".
        task (TaskType, optional): Task type. Defaults to TaskType.SEGMENTATION.
        image_metrics (str | list[str] | None, optional): Image metrics to be used for evaluation.
            Defaults to None.
        pixel_metrics (str | list[str] | None, optional): Pixel metrics to be used for evaluation.
            Defaults to None.
        visualization (DictConfig | None, optional): Visualization parameters. Defaults to None.
        **kwargs: PyTorch Lightning Trainer arguments.
    """

    def __init__(
        self,
        callbacks: list[Callback] | None = None,
        normalization: NormalizationMethod | DictConfig | Callback | str = NormalizationMethod.MIN_MAX,
        threshold: BaseThreshold
        | tuple[BaseThreshold, BaseThreshold]
        | DictConfig
        | ListConfig
        | str = "F1AdaptiveThreshold",
        task: TaskType = TaskType.SEGMENTATION,
        image_metrics: str | list[str] | None = None,
        pixel_metrics: str | list[str] | None = None,
        visualization: DictConfig | None = None,
        **kwargs,
    ) -> None:
        if callbacks is None:
            callbacks = []

        self._cache = _TrainerArgumentsCache(callbacks=[*callbacks], **kwargs)
        self.normalization = normalization
        self.threshold = threshold
        self.task = task
        self.image_metric_names = image_metrics
        self.pixel_metric_names = pixel_metrics
        self.visualization = visualization

        self._trainer: Trainer | None = None

    @property
    def trainer(self) -> Trainer:
        """Property to get the trainer.

        Raises:
            UnassignedError: When the trainer is not assigned yet.

        Returns:
            Trainer: Lightning Trainer.
        """
        if not self._trainer:
            msg = "``self.trainer`` is not assigned yet."
            raise UnassignedError(msg)
        return self._trainer

    def _setup_trainer(self, model: AnomalyModule) -> None:
        """Instantiate the trainer based on the model parameters."""
        if self._cache.requires_update(model) or self._trainer is None:
            self._cache.update(model)
            self._trainer = Trainer(**self._cache.args)
            # Callbacks need to be setup later as they depend on default_root_dir from the trainer
            self._setup_anomalib_callbacks()

    def _setup_anomalib_callbacks(self) -> None:
        """Set up callbacks for the trainer."""
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
                task=self.task,
                image_save_path=image_save_path,
                **self.visualization,
            )

        self.trainer.callbacks = _CallbackConnector._reorder_callbacks(  # noqa: SLF001
            self.trainer.callbacks + _callbacks,
        )

    def fit(
        self,
        model: AnomalyModule,
        train_dataloaders: TRAIN_DATALOADERS | LightningDataModule | None = None,
        val_dataloaders: EVAL_DATALOADERS | None = None,
        datamodule: LightningDataModule | None = None,
        ckpt_path: str | None = None,
    ) -> None:
        """Fit the model using the trainer.

        Args:
            model (AnomalyModule): Model to be trained.
            train_dataloaders (TRAIN_DATALOADERS | LightningDataModule | None, optional): Train dataloaders.
                Defaults to None.
            val_dataloaders (EVAL_DATALOADERS | None, optional): Validation dataloaders.
                Defaults to None.
            datamodule (LightningDataModule | None, optional): Lightning datamodule.
                If provided, dataloaders will be instantiated from this.
                Defaults to None.
            ckpt_path (str | None, optional): Checkpoint path. If provided, the model will be loaded from this path.
                Defaults to None.

        CLI Usage:
            1. you can pick a model, and you can run through the MVTec dataset.
                ```python
                anomalib fit --model anomalib.models.Padim
                ```
            2. Of course, you can override the various values with commands.
                ```python
                anomalib fit --model anomalib.models.Padim --data <CONFIG | CLASS_PATH_OR_NAME> --trainer.max_epochs 3
                ```
            4. If you have a ready configuration file, run it like this.
                ```python
                anomalib fit --config <config_file_path>
                ```
        """
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
        """Validate the model using the trainer.

        Args:
            model (AnomalyModule | None, optional): Model to be validated.
                Defaults to None.
            dataloaders (EVAL_DATALOADERS | LightningDataModule | None, optional): Dataloaders to be used for
                validation.
                Defaults to None.
            ckpt_path (str | None, optional): Checkpoint path. If provided, the model will be loaded from this path.
                Defaults to None.
            verbose (bool, optional): Boolean to print the validation results.
                Defaults to True.
            datamodule (LightningDataModule | None, optional): A :class:`~lightning.pytorch.core.datamodule
                LightningDataModule` that defines the
                :class:`~lightning.pytorch.core.hooks.DataHooks.val_dataloader` hook.
                Defaults to None.

        Returns:
            _EVALUATE_OUTPUT | None: Validation results.

        CLI Usage:
            1. you can pick a model.
                ```python
                anomalib validate --model anomalib.models.Padim
                ```
            2. Of course, you can override the various values with commands.
                ```python
                anomalib validate --model anomalib.models.Padim --data <CONFIG | CLASS_PATH_OR_NAME>
                ```
            4. If you have a ready configuration file, run it like this.
                ```python
                anomalib validate --config <config_file_path>
                ```
        """
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
        """Test the model using the trainer.

        Args:
            model (AnomalyModule | None, optional):
                The model to be tested.
                Defaults to None.
            dataloaders (EVAL_DATALOADERS | LightningDataModule | None, optional):
                An iterable or collection of iterables specifying test samples.
                Defaults to None.
            ckpt_path (str | None, optional):
                Either ``"best"``, ``"last"``, ``"hpc"`` or path to the checkpoint you wish to test.
                If ``None`` and the model instance was passed, use the current weights.
                Otherwise, the best model checkpoint from the previous ``trainer.fit`` call will be loaded
                if a checkpoint callback is configured.
                Defaults to None.
            verbose (bool, optional):
                If True, prints the test results.
                Defaults to True.
            datamodule (LightningDataModule | None, optional):
                A :class:`~lightning.pytorch.core.datamodule.LightningDataModule` that defines
                the :class:`~lightning.pytorch.core.hooks.DataHooks.test_dataloader` hook.
                Defaults to None.

        Returns:
            _EVALUATE_OUTPUT: A List of dictionaries containing the test results. 1 dict per dataloader.

        CLI Usage:
            1. you can pick a model.
                ```python
                anomalib test --model anomalib.models.Padim
                ```
            2. Of course, you can override the various values with commands.
                ```python
                anomalib test --model anomalib.models.Padim --data <CONFIG | CLASS_PATH_OR_NAME>
                ```
            4. If you have a ready configuration file, run it like this.
                ```python
                anomalib test --config <config_file_path>
                ```
        """
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
        """Predict using the model using the trainer.

        Args:
            model (AnomalyModule | None, optional):
                Model to be used for prediction.
                Defaults to None.
            dataloaders (EVAL_DATALOADERS | LightningDataModule | None, optional):
                An iterable or collection of iterables specifying predict samples.
                Defaults to None.
            datamodule (LightningDataModule | None, optional):
                A :class:`~lightning.pytorch.core.datamodule.LightningDataModule` that defines
                the :class:`~lightning.pytorch.core.hooks.DataHooks.predict_dataloader` hook.
                Defaults to None.
            return_predictions (bool | None, optional):
                Whether to return predictions.
                ``True`` by default except when an accelerator that spawns processes is used (not supported).
                Defaults to None.
            ckpt_path (str | None, optional):
                Either ``"best"``, ``"last"``, ``"hpc"`` or path to the checkpoint you wish to predict.
                If ``None`` and the model instance was passed, use the current weights.
                Otherwise, the best model checkpoint from the previous ``trainer.fit`` call will be loaded
                if a checkpoint callback is configured.
                Defaults to None.

        Returns:
            _PREDICT_OUTPUT | None: Predictions.

        CLI Usage:
            1. you can pick a model.
                ```python
                anomalib predict --model anomalib.models.Padim
                ```
            2. Of course, you can override the various values with commands.
                ```python
                anomalib predict --model anomalib.models.Padim --data <CONFIG | CLASS_PATH_OR_NAME>
                ```
            4. If you have a ready configuration file, run it like this.
                ```python
                anomalib predict --config <config_file_path> --return_predictions
                ```
        """
        if model:
            self._setup_trainer(model)
        return self.trainer.predict(model, dataloaders, datamodule, return_predictions, ckpt_path)
