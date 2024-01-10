"""Implements custom trainer for Anomalib."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Any

import albumentations as A  # noqa: N812
from jsonargparse import Namespace
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.trainer.connectors.callback_connector import _CallbackConnector
from lightning.pytorch.utilities.types import _EVALUATE_OUTPUT, _PREDICT_OUTPUT, EVAL_DATALOADERS, TRAIN_DATALOADERS
from omegaconf import DictConfig, ListConfig
from torch.utils.data import DataLoader, Dataset

from anomalib import TaskType
from anomalib.callbacks import get_visualization_callbacks
from anomalib.callbacks.metrics import _MetricsCallback
from anomalib.callbacks.normalization import get_normalization_callback
from anomalib.callbacks.post_processor import _PostProcessorCallback
from anomalib.callbacks.thresholding import _ThresholdCallback
from anomalib.data import AnomalibDataModule, AnomalibDataset, PredictDataset
from anomalib.deploy.export import ExportType, export_to_onnx, export_to_openvino, export_to_torch
from anomalib.metrics.threshold import BaseThreshold
from anomalib.models import AnomalyModule
from anomalib.utils.normalization import NormalizationMethod

logger = logging.getLogger(__name__)


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
                logger.info(
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

    .. note::

        Refer to PyTorch Lightning's Trainer for a list of parameters for
        details on other Trainer parameters.

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
        | list[dict[str, str | float]]
        | str = "F1AdaptiveThreshold",
        task: TaskType = TaskType.SEGMENTATION,
        image_metrics: str | list[str] | None = None,
        pixel_metrics: str | list[str] | None = None,
        visualization: DictConfig | dict[str, Any] | Namespace | None = None,
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
            if image_save_path is None or image_save_path == "":
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
        train_dataloaders: TRAIN_DATALOADERS | AnomalibDataModule | None = None,
        val_dataloaders: EVAL_DATALOADERS | None = None,
        datamodule: AnomalibDataModule | None = None,
        ckpt_path: str | None = None,
    ) -> None:
        """Fit the model using the trainer.

        Args:
            model (AnomalyModule): Model to be trained.
            train_dataloaders (TRAIN_DATALOADERS | AnomalibDataModule | None, optional): Train dataloaders.
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
        self._setup_trainer(model)
        self._setup_dataset_task(train_dataloaders, val_dataloaders, datamodule)
        self.trainer.fit(model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)

    def validate(
        self,
        model: AnomalyModule | None = None,
        dataloaders: EVAL_DATALOADERS | AnomalibDataModule | None = None,
        ckpt_path: str | None = None,
        verbose: bool = True,
        datamodule: AnomalibDataModule | None = None,
    ) -> _EVALUATE_OUTPUT | None:
        """Validate the model using the trainer.

        Args:
            model (AnomalyModule | None, optional): Model to be validated.
                Defaults to None.
            dataloaders (EVAL_DATALOADERS | AnomalibDataModule | None, optional): Dataloaders to be used for
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
        if model:
            self._setup_trainer(model)
        self._setup_dataset_task(dataloaders)
        return self.trainer.validate(model, dataloaders, ckpt_path, verbose, datamodule)

    def test(
        self,
        model: AnomalyModule | None = None,
        dataloaders: EVAL_DATALOADERS | AnomalibDataModule | None = None,
        ckpt_path: str | None = None,
        verbose: bool = True,
        datamodule: AnomalibDataModule | None = None,
    ) -> _EVALUATE_OUTPUT:
        """Test the model using the trainer.

        Args:
            model (AnomalyModule | None, optional):
                The model to be tested.
                Defaults to None.
            dataloaders (EVAL_DATALOADERS | AnomalibDataModule | None, optional):
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
        self._setup_dataset_task(dataloaders)
        return self.trainer.test(model, dataloaders, ckpt_path, verbose, datamodule)

    def predict(
        self,
        model: AnomalyModule | None = None,
        dataloaders: EVAL_DATALOADERS | AnomalibDataModule | None = None,
        datamodule: AnomalibDataModule | Dataset | PredictDataset | None = None,
        return_predictions: bool | None = None,
        ckpt_path: str | None = None,
    ) -> _PREDICT_OUTPUT | None:
        """Predict using the model using the trainer.

        Args:
            model (AnomalyModule | None, optional):
                Model to be used for prediction.
                Defaults to None.
            dataloaders (EVAL_DATALOADERS | AnomalibDataModule | None, optional):
                An iterable or collection of iterables specifying predict samples.
                Defaults to None.
            datamodule (AnomalibDataModule | None, optional):
                A :class:`~lightning.pytorch.core.datamodule.AnomalibDataModule` that defines
                the :class:`~lightning.pytorch.core.hooks.DataHooks.predict_dataloader` hook.
                The datamodule can also be a dataset that will be wrapped in a torch Dataloader.
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
        """
        if model:
            self._setup_trainer(model)

        if not ckpt_path:
            logger.warning("ckpt_path is not provided. Model weights will not be loaded.")

        # Handle the instance when a dataset is passed to the predict method
        if datamodule is not None and isinstance(datamodule, Dataset):
            dataloader = DataLoader(datamodule)
            datamodule = None
            if dataloaders is None:
                dataloaders = dataloader
            elif isinstance(dataloaders, DataLoader):
                dataloaders = [dataloaders, dataloader]
            elif isinstance(dataloaders, list):  # dataloader is a list
                dataloaders.append(dataloader)
            else:
                msg = f"Unknown type for dataloaders {type(dataloaders)}"
                raise TypeError(msg)

        self._setup_dataset_task(dataloaders, datamodule)

        return self.trainer.predict(model, dataloaders, datamodule, return_predictions, ckpt_path)

    def train(
        self,
        model: AnomalyModule,
        train_dataloaders: TRAIN_DATALOADERS | AnomalibDataModule | None = None,
        val_dataloaders: EVAL_DATALOADERS | None = None,
        test_dataloaders: EVAL_DATALOADERS | None = None,
        datamodule: AnomalibDataModule | None = None,
        ckpt_path: str | None = None,
    ) -> _EVALUATE_OUTPUT:
        """Fits the model and then calls test on it.

        Args:
            model (AnomalyModule): Model to be trained.
            train_dataloaders (TRAIN_DATALOADERS | AnomalibDataModule | None, optional): Train dataloaders.
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
                anomalib test --model anomalib.models.Padim --data <CONFIG | CLASS_PATH_OR_NAME> --trainer.max_epochs 3
                ```
            4. If you have a ready configuration file, run it like this.
                ```python
                anomalib test --config <config_file_path>
                ```
        """
        self._setup_trainer(model)
        self._setup_dataset_task(train_dataloaders, val_dataloaders, test_dataloaders, datamodule)
        self.trainer.fit(model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)
        self.trainer.test(model, test_dataloaders, ckpt_path=ckpt_path, datamodule=datamodule)

    def export(
        self,
        model: AnomalyModule,
        export_type: ExportType,
        export_root: str | Path | None = None,
        transform: dict[str, Any] | A.Compose | str | Path | None = None,
        datamodule: AnomalibDataModule | None = None,
        dataset: AnomalibDataset | None = None,
        input_size: tuple[int, int] | None = None,
        ov_args: dict[str, Any] | None = None,
        ckpt_path: str | None = None,
    ) -> Path | None:
        """Export the model in the specified format.

        Args:
            model (AnomalyModule): Trained model.
            export_type (ExportType): Export type.
            export_root (str | Path | None, optional): Path to the output directory. If it is not set, the model is
                exported to trainer.default_root_dir. Defaults to None.
            transform (dict[str, Any] | A.Compose | str | Path | None, optional): Transform config. Can either be a
                path to a file containing the transform config or can be an object. The file or object should follow
                Albumentation's format. If not provided, it takes the transform from datamodule or dataset. Datamodule
                or Dataset should be provided if transforms is not set. Defaults to None.
            datamodule (AnomalibDataModule | None, optional): Datamodule from which transforms is loaded.
                This optional. Defaults to None.
            dataset (AnomalibDataset | None, optional): Dataset from which the transforms is loaded.
                 is optional. Defaults to None.
            input_size (tuple[int, int] | None, optional): This is required only if the model is exported to ONNX and
                OpenVINO format. Defaults to None.
            ov_args (dict[str, Any] | None, optional): This is optional and used only for OpenVINO's model optimizer.
                Defaults to None.
            ckpt_path (str | None): Checkpoint path. If provided, the model will be loaded from this path.

        Returns:
            Path: Path to the exported model.

        Raises:
            ValueError: If Dataset, Datamodule, and transform are not provided.
            TypeError: If path to the transform file is not a string or Path.

        CLI Usage:
            TODO(ashwinvaidya17)
        """
        self._setup_trainer(model)
        self._setup_dataset_task(datamodule, dataset)
        if ckpt_path:
            model = model.__class__.load_from_checkpoint(ckpt_path)

        if transform is None:
            if datamodule:
                transform = datamodule.test_data.transform
            elif dataset:
                transform = dataset.transform
            else:
                logger.exception("Either datamodule or dataset must be provided if transform is None.")
                raise ValueError
        elif isinstance(transform, str | Path):
            transform = A.load(filepath=transform, data_format="yaml")
        else:
            logger.exception(f"Unknown type {type(transform)} for transform.")
            raise TypeError

        if export_root is None:
            export_root = Path(self.trainer.default_root_dir)

        exported_model_path: Path | None = None
        if export_type == ExportType.TORCH:
            exported_model_path = export_to_torch(
                model=model,
                export_root=export_root,
                transform=transform,
                task=self.task,
            )
        elif export_type == ExportType.ONNX:
            assert input_size is not None, "input_size must be provided for ONNX export type."
            exported_model_path = export_to_onnx(
                model=model,
                input_size=input_size,
                export_root=export_root,
                transform=transform,
                task=self.task,
            )
        elif export_type == ExportType.OPENVINO:
            assert input_size is not None, "input_size must be provided for OpenVINO export type."
            exported_model_path = export_to_openvino(
                model=model,
                input_size=input_size,
                export_root=export_root,
                transform=transform,
                task=self.task,
                ov_args=ov_args,
            )
        else:
            logging.error(f"Export type {export_type} is not supported yet.")

        if exported_model_path:
            logging.info(f"Exported to {exported_model_path}")
        return exported_model_path
