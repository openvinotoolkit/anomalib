"""Implements custom trainer for Anomalib."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.utilities.types import _EVALUATE_OUTPUT, _PREDICT_OUTPUT, EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Metric
from torchvision.transforms.v2 import Transform

from anomalib import LearningType, TaskType
from anomalib.callbacks.checkpoint import ModelCheckpoint
from anomalib.callbacks.metrics import _MetricsCallback
from anomalib.callbacks.normalization import get_normalization_callback
from anomalib.callbacks.normalization.base import NormalizationCallback
from anomalib.callbacks.post_processor import _PostProcessorCallback
from anomalib.callbacks.thresholding import _ThresholdCallback
from anomalib.callbacks.timer import TimerCallback
from anomalib.callbacks.visualizer import _VisualizationCallback
from anomalib.data import AnomalibDataModule, AnomalibDataset, PredictDataset
from anomalib.deploy import CompressionType, ExportType
from anomalib.models import AnomalyModule
from anomalib.utils.normalization import NormalizationMethod
from anomalib.utils.path import create_versioned_dir
from anomalib.utils.types import NORMALIZATION, THRESHOLD
from anomalib.utils.visualization import ImageVisualizer

logger = logging.getLogger(__name__)


class UnassignedError(Exception):
    """Unassigned error."""


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
                logger.info(
                    f"Overriding {key} from {self._cached_args[key]} with {value} for {model.__class__.__name__}",
                )
            self._cached_args[key] = value

    def requires_update(self, model: AnomalyModule) -> bool:
        return any(self._cached_args.get(key, None) != value for key, value in model.trainer_arguments.items())

    @property
    def args(self) -> dict[str, Any]:
        return self._cached_args


class Engine:
    """Anomalib Engine.

    .. note::

        Refer to PyTorch Lightning's Trainer for a list of parameters for
        details on other Trainer parameters.

    Args:
        callbacks (list[Callback]): Add a callback or list of callbacks.
        normalization (NORMALIZATION, optional): Normalization method.
            Defaults to NormalizationMethod.MIN_MAX.
        threshold (THRESHOLD):
            Thresholding method. Defaults to "F1AdaptiveThreshold".
        task (TaskType, optional): Task type. Defaults to TaskType.SEGMENTATION.
        image_metrics (list[str] | str | dict[str, dict[str, Any]] | None, optional): Image metrics to be used for
            evaluation. Defaults to None.
        pixel_metrics (list[str] | str | dict[str, dict[str, Any]] | None, optional): Pixel metrics to be used for
            evaluation. Defaults to None.
        default_root_dir (str, optional): Default root directory for the trainer.
            The results will be saved in this directory.
            Defaults to ``results``.
        **kwargs: PyTorch Lightning Trainer arguments.
    """

    def __init__(
        self,
        callbacks: list[Callback] | None = None,
        normalization: NORMALIZATION = NormalizationMethod.MIN_MAX,
        threshold: THRESHOLD = "F1AdaptiveThreshold",
        task: TaskType | str = TaskType.SEGMENTATION,
        image_metrics: list[str] | str | dict[str, dict[str, Any]] | None = None,
        pixel_metrics: list[str] | str | dict[str, dict[str, Any]] | None = None,
        logger: Logger | Iterable[Logger] | bool | None = None,
        default_root_dir: str | Path = "results",
        **kwargs,
    ) -> None:
        # TODO(ashwinvaidya17): Add model argument to engine constructor
        # https://github.com/openvinotoolkit/anomalib/issues/1639
        if callbacks is None:
            callbacks = []

        # Cache the Lightning Trainer arguments.
        logger = False if logger is None else logger
        self._cache = _TrainerArgumentsCache(
            callbacks=[*callbacks],
            logger=logger,
            default_root_dir=Path(default_root_dir),
            **kwargs,
        )

        self.normalization = normalization
        self.threshold = threshold
        self.task = TaskType(task)
        self.image_metric_names = image_metrics if image_metrics else ["AUROC", "F1Score"]

        # pixel metrics are only used for segmentation tasks.
        self.pixel_metric_names = None
        if self.task == TaskType.SEGMENTATION:
            self.pixel_metric_names = pixel_metrics if pixel_metrics is not None else ["AUROC", "F1Score"]

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

    @property
    def model(self) -> AnomalyModule:
        """Property to get the model.

        Raises:
            UnassignedError: When the model is not assigned yet.

        Returns:
            AnomalyModule: Anomaly model.
        """
        if not self.trainer.lightning_module:
            msg = "Trainer does not have a model assigned yet."
            raise UnassignedError(msg)
        return self.trainer.lightning_module

    @property
    def normalization_callback(self) -> NormalizationCallback | None:
        """The ``NormalizationCallback`` callback in the trainer.callbacks list, or ``None`` if it doesn't exist.

        Returns:
            NormalizationCallback | None: Normalization callback, if available.

        Raises:
            ValueError: If there are multiple normalization callbacks.
        """
        callbacks = [callback for callback in self.trainer.callbacks if isinstance(callback, NormalizationCallback)]
        if len(callbacks) > 1:
            msg = (
                f"Trainer can only have one normalization callback but multiple found: {callbacks}. "
                "Please check your configuration. Exiting to avoid unexpected behavior."
            )
            raise ValueError(msg)
        return callbacks[0] if len(callbacks) > 0 else None

    @property
    def threshold_callback(self) -> _ThresholdCallback | None:
        """The ``ThresholdCallback`` callback in the trainer.callbacks list, or ``None`` if it doesn't exist.

        Returns:
            _ThresholdCallback | None: Threshold callback, if available.

        Raises:
            ValueError: If there are multiple threshold callbacks.
        """
        callbacks = [callback for callback in self.trainer.callbacks if isinstance(callback, _ThresholdCallback)]
        if len(callbacks) > 1:
            msg = (
                f"Trainer can only have one thresholding callback but multiple found: {callbacks}. "
                "Please check your configuration. Exiting to avoid unexpected behavior."
            )
            raise ValueError(msg)
        return callbacks[0] if len(callbacks) > 0 else None

    @property
    def checkpoint_callback(self) -> ModelCheckpoint | None:
        """The ``ModelCheckpoint`` callback in the trainer.callbacks list, or ``None`` if it doesn't exist.

        Returns:
            ModelCheckpoint | None: ModelCheckpoint callback, if available.
        """
        if self._trainer is None:
            return None
        return self.trainer.checkpoint_callback

    @property
    def best_model_path(self) -> str | None:
        """The path to the best model checkpoint.

        Returns:
            str: Path to the best model checkpoint.
        """
        if self.checkpoint_callback is None:
            return None
        return self.checkpoint_callback.best_model_path

    def _setup_workspace(
        self,
        model: AnomalyModule,
        train_dataloaders: TRAIN_DATALOADERS | None = None,
        val_dataloaders: EVAL_DATALOADERS | None = None,
        test_dataloaders: EVAL_DATALOADERS | None = None,
        datamodule: AnomalibDataModule | None = None,
        dataset: AnomalibDataset | None = None,
        versioned_dir: bool = False,
    ) -> None:
        """Setup the workspace for the model.

        This method sets up the default root directory for the model based on
        the model name, dataset name, and category. Model checkpoints, logs, and
        other artifacts will be saved in this directory.

        Args:
            model (AnomalyModule): Input model.
            train_dataloaders (TRAIN_DATALOADERS | None, optional): Train dataloaders.
                Defaults to ``None``.
            val_dataloaders (EVAL_DATALOADERS | None, optional): Validation dataloaders.
                Defaults to ``None``.
            test_dataloaders (EVAL_DATALOADERS | None, optional): Test dataloaders.
                Defaults to ``None``.
            datamodule (AnomalibDataModule | None, optional): Lightning datamodule.
                Defaults to ``None``.
            dataset (AnomalibDataset | None, optional): Anomalib dataset.
                Defaults to ``None``.
            versioned_dir (bool, optional): Whether to create a versioned directory.
                Defaults to ``True``.

        Raises:
            TypeError: If the dataloader type is unknown.
        """
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        # 1. Get the dataset name and category from the dataloaders, datamodule, or dataset.
        dataset_name: str = ""
        category: str | None = None

        # Check datamodule and dataset directly
        if datamodule is not None:
            dataset_name = datamodule.name
            category = datamodule.category
        elif dataset is not None:
            dataset_name = dataset.name
            category = dataset.category

        # Check dataloaders if dataset_name and category are not set
        dataloaders = [train_dataloaders, val_dataloaders, test_dataloaders]
        if not dataset_name or category is None:
            for dataloader in dataloaders:
                if dataloader is not None:
                    if hasattr(dataloader, "train_data"):
                        dataset_name = getattr(dataloader.train_data, "name", "")
                        category = getattr(dataloader.train_data, "category", "")
                        break
                    if dataset_name and category is not None:
                        break

        # Check if category is None and set it to empty string
        category = category if category is not None else ""

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        # 2. Update the default root directory
        root_dir = Path(self._cache.args["default_root_dir"]) / model.name / dataset_name / category
        self._cache.args["default_root_dir"] = create_versioned_dir(root_dir) if versioned_dir else root_dir / "latest"

    def _setup_trainer(self, model: AnomalyModule) -> None:
        """Instantiate the trainer based on the model parameters."""
        # Check if the cache requires an update
        if self._cache.requires_update(model):
            self._cache.update(model)

        # Setup anomalib callbacks to be used with the trainer
        self._setup_anomalib_callbacks()

        # Temporarily set devices to 1 to avoid issues with multiple processes
        self._cache.args["devices"] = 1

        # Instantiate the trainer if it is not already instantiated
        if self._trainer is None:
            self._trainer = Trainer(**self._cache.args)

    def _setup_dataset_task(
        self,
        *dataloaders: EVAL_DATALOADERS | TRAIN_DATALOADERS | AnomalibDataModule | None,
    ) -> None:
        """Override the dataloader task with the task passed to the Engine.

        Args:
            dataloaders (TRAIN_DATALOADERS | EVAL_DATALOADERS): Dataloaders to be used for training or evaluation.
        """
        for dataloader in dataloaders:
            if dataloader is not None and isinstance(dataloader, AnomalibDataModule):
                for attribute in ("train_data", "val_data", "test_data"):
                    if hasattr(dataloader, attribute):
                        data: AnomalibDataset = getattr(dataloader, attribute)
                        if data.task != self.task:
                            logger.info(
                                f"Overriding task from {data.task} with {self.task} for {dataloader.__class__}",
                            )
                            data.task = self.task

    @staticmethod
    def _setup_transform(
        model: AnomalyModule,
        datamodule: AnomalibDataModule | None = None,
        dataloaders: EVAL_DATALOADERS | TRAIN_DATALOADERS | None = None,
        ckpt_path: Path | str | None = None,
    ) -> None:
        """Implements the logic for setting the transform at the start of each run.

        Any transform passed explicitly to the datamodule takes precedence. Otherwise, if a checkpoint path is provided,
        we can load the transform from the checkpoint. If no transform is provided, we use the default transform from
        the model.

        Args:
            model (AnomalyModule): The model to assign the transform to.
            datamodule (AnomalibDataModule | None): The datamodule to assign the transform from.
                defaults to ``None``.
            dataloaders (EVAL_DATALOADERS | TRAIN_DATALOADERS | None): Dataloaders to assign the transform to.
                defaults to ``None``.
            ckpt_path (str): The path to the checkpoint.
                defaults to ``None``.

        Returns:
            Transform: The transform loaded from the checkpoint.
        """
        if isinstance(dataloaders, DataLoader):
            dataloaders = [dataloaders]

        # get transform
        if datamodule and datamodule.transform:
            # a transform passed explicitly to the datamodule takes precedence
            transform = datamodule.transform
        elif dataloaders and any(getattr(dl.dataset, "transform", None) for dl in dataloaders):
            # if dataloaders are provided, we use the transform from the first dataloader that has a transform
            transform = next(dl.dataset.transform for dl in dataloaders if getattr(dl.dataset, "transform", None))
        elif ckpt_path is not None:
            # if a checkpoint path is provided, we can load the transform from the checkpoint
            checkpoint = torch.load(ckpt_path, map_location=model.device)
            transform = checkpoint["transform"]
        elif model.transform is None:
            # if no transform is provided, we use the default transform from the model
            image_size = datamodule.image_size if datamodule else None
            transform = model.configure_transforms(image_size)
        else:
            transform = model.transform

        # update transform in model
        model.set_transform(transform)
        # The dataloaders don't have access to the trainer and/or model, so we need to set the transforms manually
        if dataloaders:
            for dataloader in dataloaders:
                if not getattr(dataloader.dataset, "transform", None):
                    dataloader.dataset.transform = transform

    def _setup_anomalib_callbacks(self) -> None:
        """Set up callbacks for the trainer."""
        _callbacks: list[Callback] = []

        # Add ModelCheckpoint if it is not in the callbacks list.
        has_checkpoint_callback = any(isinstance(c, ModelCheckpoint) for c in self._cache.args["callbacks"])
        if has_checkpoint_callback is False:
            _callbacks.append(
                ModelCheckpoint(
                    dirpath=self._cache.args["default_root_dir"] / "weights" / "lightning",
                    filename="model",
                    auto_insert_metric_name=False,
                ),
            )

        # Add the post-processor callbacks.
        _callbacks.append(_PostProcessorCallback())

        # Add the the normalization callback.
        normalization_callback = get_normalization_callback(self.normalization)
        if normalization_callback is not None:
            _callbacks.append(normalization_callback)

        # Add the thresholding and metrics callbacks.
        _callbacks.append(_ThresholdCallback(self.threshold))
        _callbacks.append(_MetricsCallback(self.task, self.image_metric_names, self.pixel_metric_names))

        _callbacks.append(
            _VisualizationCallback(
                visualizers=ImageVisualizer(task=self.task, normalize=self.normalization == NormalizationMethod.NONE),
                save=True,
                root=self._cache.args["default_root_dir"] / "images",
            ),
        )

        _callbacks.append(TimerCallback())

        # Combine the callbacks, and update the trainer callbacks.
        self._cache.args["callbacks"] = _callbacks + self._cache.args["callbacks"]

    def _should_run_validation(
        self,
        model: AnomalyModule,
        dataloaders: EVAL_DATALOADERS | None,
        datamodule: AnomalibDataModule | None,
        ckpt_path: str | Path | None,
    ) -> bool:
        """Check if we need to run validation to collect normalization statistics and thresholds.

        If a checkpoint path is provided, we don't need to run validation because we can load the model from the
        checkpoint and use the normalization metrics and thresholds from the checkpoint.

        We need to run validation if the model is configured with normalization enabled, but no normalization metrics
        have been collected yet. Similarly, we need to run validation if the model is configured with adaptive
        thresholding enabled, but no thresholds have been computed yet.

        We can only run validation if we have validation data available, so we check if the dataloaders or datamodule
        are available. If neither is available, we can't run validation.

        Args:
            model (AnomalyModule): Model passed to the entrypoint.
            dataloaders (EVAL_DATALOADERS | None): Dataloaders passed to the entrypoint.
            datamodule (AnomalibDataModule | None): Lightning datamodule passed to the entrypoint.
            ckpt_path (str | Path | None): Checkpoint path passed to the entrypoint.

        Returns:
            bool: Whether it is needed to run a validation sequence.
        """
        # validation before predict is only necessary for zero-/few-shot models
        if model.learning_type not in {LearningType.ZERO_SHOT, LearningType.FEW_SHOT}:
            return False
        # check if a checkpoint path is provided
        if ckpt_path is not None:
            return False
        # check if the model needs to be validated
        needs_normalization = self.normalization_callback is not None and not hasattr(model, "normalization_metrics")
        needs_thresholding = self.threshold_callback is not None and not hasattr(model, "image_threshold")
        # check if the model can be validated (i.e. validation data is available)
        return (needs_normalization or needs_thresholding) and (dataloaders is not None or datamodule is not None)

    def fit(
        self,
        model: AnomalyModule,
        train_dataloaders: TRAIN_DATALOADERS | None = None,
        val_dataloaders: EVAL_DATALOADERS | None = None,
        datamodule: AnomalibDataModule | None = None,
        ckpt_path: str | Path | None = None,
    ) -> None:
        """Fit the model using the trainer.

        Args:
            model (AnomalyModule): Model to be trained.
            train_dataloaders (TRAIN_DATALOADERS | None, optional): Train dataloaders.
                Defaults to None.
            val_dataloaders (EVAL_DATALOADERS | None, optional): Validation dataloaders.
                Defaults to None.
            datamodule (AnomalibDataModule | None, optional): Lightning datamodule.
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
        if ckpt_path:
            ckpt_path = Path(ckpt_path).resolve()

        self._setup_workspace(
            model=model,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
            datamodule=datamodule,
            versioned_dir=True,
        )
        self._setup_trainer(model)
        self._setup_dataset_task(train_dataloaders, val_dataloaders, datamodule)
        self._setup_transform(model, datamodule=datamodule, ckpt_path=ckpt_path)
        if model.learning_type in {LearningType.ZERO_SHOT, LearningType.FEW_SHOT}:
            # if the model is zero-shot or few-shot, we only need to run validate for normalization and thresholding
            self.trainer.validate(model, val_dataloaders, datamodule=datamodule, ckpt_path=ckpt_path)
        else:
            self.trainer.fit(model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)

    def validate(
        self,
        model: AnomalyModule | None = None,
        dataloaders: EVAL_DATALOADERS | None = None,
        ckpt_path: str | Path | None = None,
        verbose: bool = True,
        datamodule: AnomalibDataModule | None = None,
    ) -> _EVALUATE_OUTPUT | None:
        """Validate the model using the trainer.

        Args:
            model (AnomalyModule | None, optional): Model to be validated.
                Defaults to None.
            dataloaders (EVAL_DATALOADERS | None, optional): Dataloaders to be used for
                validation.
                Defaults to None.
            ckpt_path (str | None, optional): Checkpoint path. If provided, the model will be loaded from this path.
                Defaults to None.
            verbose (bool, optional): Boolean to print the validation results.
                Defaults to True.
            datamodule (AnomalibDataModule | None, optional): A :class:`~lightning.pytorch.core.datamodule
                AnomalibDataModule` that defines the
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
        if ckpt_path:
            ckpt_path = Path(ckpt_path).resolve()
        if model:
            self._setup_trainer(model)
        self._setup_dataset_task(dataloaders)
        self._setup_transform(model or self.model, datamodule=datamodule, ckpt_path=ckpt_path)
        return self.trainer.validate(model, dataloaders, ckpt_path, verbose, datamodule)

    def test(
        self,
        model: AnomalyModule | None = None,
        dataloaders: EVAL_DATALOADERS | None = None,
        ckpt_path: str | Path | None = None,
        verbose: bool = True,
        datamodule: AnomalibDataModule | None = None,
    ) -> _EVALUATE_OUTPUT:
        """Test the model using the trainer.

        Sets up the trainer and the dataset task if not already set up. Then validates the model if needed and
        finally tests the model.

        Args:
            model (AnomalyModule | None, optional):
                The model to be tested.
                Defaults to None.
            dataloaders (EVAL_DATALOADERS | None, optional):
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
            datamodule (AnomalibDataModule | None, optional):
                A :class:`~lightning.pytorch.core.datamodule.AnomalibDataModule` that defines
                the :class:`~lightning.pytorch.core.hooks.DataHooks.test_dataloader` hook.
                Defaults to None.

        Returns:
            _EVALUATE_OUTPUT: A List of dictionaries containing the test results. 1 dict per dataloader.

        Examples:
            # fit and test a one-class model
            >>> from anomalib.data import MVTec
            >>> from anomalib.models import Padim
            >>> from anomalib.engine import Engine

            >>> datamodule = MVTec()
            >>> model = Padim()
            >>> model.learning_type
            <LearningType.ONE_CLASS: 'one_class'>

            >>> engine = Engine()
            >>> engine.fit(model, datamodule=datamodule)
            >>> engine.test(model, datamodule=datamodule)

            # Test a zero-shot model
            >>> from anomalib.data import MVTec
            >>> from anomalib.models import Padim
            >>> from anomalib.engine import Engine

            >>> datamodule = MVTec(image_size=240, normalization="clip")
            >>> model = Padim()
            >>> model.learning_type
            <LearningType.ZERO_SHOT: 'zero_shot'>

            >>> engine = Engine()
            >>> engine.test(model, datamodule=datamodule)

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
        if ckpt_path:
            ckpt_path = Path(ckpt_path).resolve()

        self._setup_workspace(model=model or self.model, datamodule=datamodule, test_dataloaders=dataloaders)

        if model:
            self._setup_trainer(model)
        elif not self.model:
            msg = "`Engine.test()` requires an `AnomalyModule` when it hasn't been passed in a previous run."
            raise RuntimeError(msg)

        self._setup_dataset_task(dataloaders)
        self._setup_transform(model or self.model, datamodule=datamodule, ckpt_path=ckpt_path)
        if self._should_run_validation(model or self.model, dataloaders, datamodule, ckpt_path):
            logger.info("Running validation before testing to collect normalization metrics and/or thresholds.")
            self.trainer.validate(model, dataloaders, None, verbose=False, datamodule=datamodule)
        return self.trainer.test(model, dataloaders, ckpt_path, verbose, datamodule)

    def predict(
        self,
        model: AnomalyModule | None = None,
        dataloaders: EVAL_DATALOADERS | None = None,
        datamodule: AnomalibDataModule | None = None,
        dataset: Dataset | PredictDataset | None = None,
        return_predictions: bool | None = None,
        ckpt_path: str | Path | None = None,
        data_path: str | Path | None = None,
    ) -> _PREDICT_OUTPUT | None:
        """Predict using the model using the trainer.

        Sets up the trainer and the dataset task if not already set up. Then validates the model if needed and a
        validation dataloader is available. Finally, predicts using the model.

        Args:
            model (AnomalyModule | None, optional):
                Model to be used for prediction.
                Defaults to None.
            dataloaders (EVAL_DATALOADERS | None, optional):
                An iterable or collection of iterables specifying predict samples.
                Defaults to None.
            datamodule (AnomalibDataModule | None, optional):
                A :class:`~lightning.pytorch.core.datamodule.AnomalibDataModule` that defines
                the :class:`~lightning.pytorch.core.hooks.DataHooks.predict_dataloader` hook.
                The datamodule can also be a dataset that will be wrapped in a torch Dataloader.
                Defaults to None.
            dataset (Dataset | PredictDataset | None, optional):
                A :class:`~torch.utils.data.Dataset` or :class:`~anomalib.data.PredictDataset` that will be used
                to create a dataloader. Defaults to None.
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
            data_path (str | Path | None):
                Path to the image or folder containing images to generate predictions for.
                Defaults to None.

        Returns:
            _PREDICT_OUTPUT | None: Predictions.

        CLI Usage:
            1. you can pick a model.
                ```python
                anomalib predict --model anomalib.models.Padim
                anomalib predict --model Padim \
                                 --data datasets/MVTec/bottle/test/broken_large
                ```
            2. Of course, you can override the various values with commands.
                ```python
                anomalib predict --model anomalib.models.Padim \
                                 --data <CONFIG | CLASS_PATH_OR_NAME>
                ```
            4. If you have a ready configuration file, run it like this.
                ```python
                anomalib predict --config <config_file_path> --return_predictions
                ```
            5. You can also point to a folder with image or a single image instead of passing a dataset.
                ```python
                anomalib predict --model Padim --data <PATH_TO_IMAGE_OR_FOLDER> --ckpt_path <PATH_TO_CHECKPOINT>
                ```
        """
        if not (model or self.model):
            msg = "`Engine.predict()` requires an `AnomalyModule` when it hasn't been passed in a previous run."
            raise ValueError(msg)

        if ckpt_path:
            ckpt_path = Path(ckpt_path).resolve()

        self._setup_workspace(model=model or self.model, datamodule=datamodule, test_dataloaders=dataloaders)

        if model:
            self._setup_trainer(model)

        if not ckpt_path:
            logger.warning("ckpt_path is not provided. Model weights will not be loaded.")

        # Collect dataloaders
        if dataloaders is None:
            dataloaders = []
        elif isinstance(dataloaders, DataLoader):
            dataloaders = [dataloaders]
        elif not isinstance(dataloaders, list):
            msg = f"Unknown type for dataloaders {type(dataloaders)}"
            raise TypeError(msg)
        if dataset is not None:
            dataloaders.append(DataLoader(dataset))
        if data_path is not None:
            dataloaders.append(DataLoader(PredictDataset(data_path)))
        dataloaders = dataloaders or None

        self._setup_dataset_task(dataloaders, datamodule)
        self._setup_transform(model or self.model, datamodule=datamodule, dataloaders=dataloaders, ckpt_path=ckpt_path)

        if self._should_run_validation(model or self.model, None, datamodule, ckpt_path):
            logger.info("Running validation before predicting to collect normalization metrics and/or thresholds.")
            self.trainer.validate(
                model,
                dataloaders=None,
                ckpt_path=None,
                verbose=False,
                datamodule=datamodule,
            )

        return self.trainer.predict(model, dataloaders, datamodule, return_predictions, ckpt_path)

    def train(
        self,
        model: AnomalyModule,
        train_dataloaders: TRAIN_DATALOADERS | None = None,
        val_dataloaders: EVAL_DATALOADERS | None = None,
        test_dataloaders: EVAL_DATALOADERS | None = None,
        datamodule: AnomalibDataModule | None = None,
        ckpt_path: str | Path | None = None,
    ) -> _EVALUATE_OUTPUT:
        """Fits the model and then calls test on it.

        Args:
            model (AnomalyModule): Model to be trained.
            train_dataloaders (TRAIN_DATALOADERS | None, optional): Train dataloaders.
                Defaults to None.
            val_dataloaders (EVAL_DATALOADERS | None, optional): Validation dataloaders.
                Defaults to None.
            test_dataloaders (EVAL_DATALOADERS | None, optional): Test dataloaders.
                Defaults to None.
            datamodule (AnomalibDataModule | None, optional): Lightning datamodule.
                If provided, dataloaders will be instantiated from this.
                Defaults to None.
            ckpt_path (str | None, optional): Checkpoint path. If provided, the model will be loaded from this path.
                Defaults to None.

        CLI Usage:
            1. you can pick a model, and you can run through the MVTec dataset.
                ```python
                anomalib train --model anomalib.models.Padim --data MVTec
                ```
            2. Of course, you can override the various values with commands.
                ```python
                anomalib train --model anomalib.models.Padim --data <CONFIG | CLASS_PATH_OR_NAME> --trainer.max_epochs 3
                ```
            4. If you have a ready configuration file, run it like this.
                ```python
                anomalib train --config <config_file_path>
                ```
        """
        if ckpt_path:
            ckpt_path = Path(ckpt_path).resolve()
        self._setup_workspace(
            model,
            train_dataloaders,
            val_dataloaders,
            test_dataloaders,
            datamodule,
            versioned_dir=True,
        )
        self._setup_trainer(model)
        self._setup_dataset_task(
            train_dataloaders,
            val_dataloaders,
            test_dataloaders,
            datamodule,
        )
        self._setup_transform(model, datamodule=datamodule, ckpt_path=ckpt_path)
        if model.learning_type in {LearningType.ZERO_SHOT, LearningType.FEW_SHOT}:
            # if the model is zero-shot or few-shot, we only need to run validate for normalization and thresholding
            self.trainer.validate(model, val_dataloaders, None, verbose=False, datamodule=datamodule)
        else:
            self.trainer.fit(model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)
        self.trainer.test(model, test_dataloaders, ckpt_path=ckpt_path, datamodule=datamodule)

    def export(
        self,
        model: AnomalyModule,
        export_type: ExportType | str,
        export_root: str | Path | None = None,
        input_size: tuple[int, int] | None = None,
        transform: Transform | None = None,
        compression_type: CompressionType | None = None,
        datamodule: AnomalibDataModule | None = None,
        metric: Metric | str | None = None,
        ov_args: dict[str, Any] | None = None,
        ckpt_path: str | Path | None = None,
    ) -> Path | None:
        r"""Export the model in PyTorch, ONNX or OpenVINO format.

        Args:
            model (AnomalyModule): Trained model.
            export_type (ExportType): Export type.
            export_root (str | Path | None, optional): Path to the output directory. If it is not set, the model is
                exported to trainer.default_root_dir. Defaults to None.
            input_size (tuple[int, int] | None, optional): A statis input shape for the model, which is exported to ONNX
                and OpenVINO format. Defaults to None.
            transform (Transform | None, optional): Input transform to include in the exported model. If not provided,
                the engine will try to use the default transform from the model.
                Defaults to ``None``.
            compression_type (CompressionType | None, optional): Compression type for OpenVINO exporting only.
                Defaults to ``None``.
            datamodule (AnomalibDataModule | None, optional): Lightning datamodule.
                Must be provided if ``CompressionType.INT8_PTQ`` or `CompressionType.INT8_ACQ`` is selected
                (OpenVINO export only).
                Defaults to ``None``.
            metric (Metric | str | None, optional): Metric to measure quality loss when quantizing.
                Must be provided if ``CompressionType.INT8_ACQ`` is selected and must return higher value for better
                performance of the model (OpenVINO export only).
                Defaults to ``None``.
            ov_args (dict[str, Any] | None, optional): This is optional and used only for OpenVINO's model optimizer.
                Defaults to None.
            ckpt_path (str | Path | None): Checkpoint path. If provided, the model will be loaded from this path.

        Returns:
            Path: Path to the exported model.

        Raises:
            ValueError: If Dataset, Datamodule, and transform are not provided.
            TypeError: If path to the transform file is not a string or Path.

        CLI Usage:
            1. To export as a torch ``.pt`` file you can run the following command.
                ```python
                anomalib export --model Padim --export_type torch --ckpt_path <PATH_TO_CHECKPOINT>
                ```
            2. To export as an ONNX ``.onnx`` file you can run the following command.
                ```python
                anomalib export --model Padim --export_type onnx --ckpt_path <PATH_TO_CHECKPOINT> \
                --input_size "[256,256]"
                ```
            3. To export as an OpenVINO ``.xml`` and ``.bin`` file you can run the following command.
                ```python
                anomalib export --model Padim --export_type openvino --ckpt_path <PATH_TO_CHECKPOINT> \
                --input_size "[256,256] --compression_type FP16
                ```
            4. You can also quantize OpenVINO model with the following.
                ```python
                anomalib export --model Padim --export_type openvino --ckpt_path <PATH_TO_CHECKPOINT> \
                --input_size "[256,256]" --compression_type INT8_PTQ --data MVTec
                ```
        """
        export_type = ExportType(export_type)
        self._setup_trainer(model)
        if ckpt_path:
            ckpt_path = Path(ckpt_path).resolve()
            model = model.__class__.load_from_checkpoint(ckpt_path)

        if export_root is None:
            export_root = Path(self.trainer.default_root_dir)

        exported_model_path: Path | None = None
        if export_type == ExportType.TORCH:
            exported_model_path = model.to_torch(
                export_root=export_root,
                transform=transform,
                task=self.task,
            )
        elif export_type == ExportType.ONNX:
            exported_model_path = model.to_onnx(
                export_root=export_root,
                input_size=input_size,
                transform=transform,
                task=self.task,
            )
        elif export_type == ExportType.OPENVINO:
            exported_model_path = model.to_openvino(
                export_root=export_root,
                input_size=input_size,
                transform=transform,
                task=self.task,
                compression_type=compression_type,
                datamodule=datamodule,
                metric=metric,
                ov_args=ov_args,
            )
        else:
            logging.error(f"Export type {export_type} is not supported yet.")

        if exported_model_path:
            logging.info(f"Exported model to {exported_model_path}")
        return exported_model_path

    @classmethod
    def from_config(
        cls: type["Engine"],
        config_path: str | Path,
        **kwargs,
    ) -> tuple["Engine", AnomalyModule, AnomalibDataModule]:
        """Create an Engine instance from a configuration file.

        Args:
            config_path (str | Path): Path to the full configuration file.
            **kwargs (dict): Additional keyword arguments.

        Returns:
            tuple[Engine, AnomalyModule, AnomalibDataModule]: Engine instance.

        Example:
            The following example shows training with full configuration file:

            .. code-block:: python
                >>> config_path = "anomalib_full_config.yaml"
                >>> engine, model, datamodule = Engine.from_config(config_path=config_path)
                >>> engine.fit(datamodule=datamodule, model=model)

            The following example shows overriding the configuration file with additional keyword arguments:

            .. code-block:: python
                >>> override_kwargs = {"data.train_batch_size": 8}
                >>> engine, model, datamodule = Engine.from_config(config_path=config_path, **override_kwargs)
                >>> engine.fit(datamodule=datamodule, model=model)
        """
        from anomalib.cli.cli import AnomalibCLI

        if not Path(config_path).exists():
            msg = f"Configuration file not found: {config_path}"
            raise FileNotFoundError(msg)

        args = [
            "fit",
            "--config",
            str(config_path),
        ]
        for key, value in kwargs.items():
            args.extend([f"--{key}", str(value)])
        anomalib_cli = AnomalibCLI(
            args=args,
            run=False,
        )
        return anomalib_cli.engine, anomalib_cli.model, anomalib_cli.datamodule
