"""Base Anomaly Module for Training Task."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import logging
from abc import ABC, abstractproperty
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import albumentations as A  # noqa: N812
import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn

from anomalib import LearningType, TaskType
from anomalib.data import AnomalibDataModule, AnomalibDataset
from anomalib.metrics import AnomalibMetricCollection
from anomalib.metrics.threshold import BaseThreshold
from anomalib.utils.exceptions import try_import

if TYPE_CHECKING:
    from lightning.pytorch.callbacks import Callback
    from torch.types import Number
    from torchmetrics import Metric


logger = logging.getLogger(__name__)
if try_import("openvino"):
    from openvino.runtime import serialize
    from openvino.tools.ovc import convert_model


class ExportType(str, Enum):
    """Model export type.

    Examples:
        >>> from anomalib.models import ExportType
        >>> ExportType.ONNX
        'onnx'
        >>> ExportType.OPENVINO
        'openvino'
        >>> ExportType.TORCH
        'torch'
    """

    ONNX = "onnx"
    OPENVINO = "openvino"
    TORCH = "torch"


class AnomalyModule(pl.LightningModule, ABC):
    """AnomalyModule to train, validate, predict and test images.

    Acts as a base class for all the Anomaly Modules in the library.
    """

    def __init__(self) -> None:
        super().__init__()
        logger.info("Initializing %s model.", self.__class__.__name__)

        self.save_hyperparameters()
        self.model: nn.Module
        self.loss: nn.Module
        self.callbacks: list[Callback]

        self.image_threshold: BaseThreshold
        self.pixel_threshold: BaseThreshold

        self.normalization_metrics: Metric

        self.image_metrics: AnomalibMetricCollection
        self.pixel_metrics: AnomalibMetricCollection

    def forward(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> Any:  # noqa: ANN401
        """Perform the forward-pass by passing input tensor to the module.

        Args:
            batch (dict[str, str | torch.Tensor]): Input batch.
            *args: Arguments.
            **kwargs: Keyword arguments.

        Returns:
            Tensor: Output tensor from the model.
        """
        del args, kwargs  # These variables are not used.

        return self.model(batch)

    def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """To be implemented in the subclasses."""
        raise NotImplementedError

    def predict_step(
        self,
        batch: dict[str, str | torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> STEP_OUTPUT:
        """Step function called during :meth:`~lightning.pytorch.trainer.Trainer.predict`.

        By default, it calls :meth:`~lightning.pytorch.core.lightning.LightningModule.forward`.
        Override to add any processing logic.

        Args:
            batch (Any): Current batch
            batch_idx (int): Index of current batch
            dataloader_idx (int): Index of the current dataloader

        Return:
            Predicted output
        """
        del dataloader_idx  # These variables are not used.

        return self.validation_step(batch, batch_idx)

    def test_step(self, batch: dict[str, str | torch.Tensor], batch_idx: int, *args, **kwargs) -> STEP_OUTPUT:
        """Calls validation_step for anomaly map/score calculation.

        Args:
          batch (dict[str, str | torch.Tensor]): Input batch
          batch_idx (int): Batch index
          args: Arguments.
          kwargs: Keyword arguments.

        Returns:
          Dictionary containing images, features, true labels and masks.
          These are required in `validation_epoch_end` for feature concatenation.
        """
        del args, kwargs  # These variables are not used.

        return self.predict_step(batch, batch_idx)

    @abstractproperty
    def trainer_arguments(self) -> dict[str, Any]:
        """Arguments used to override the trainer parameters so as to train the model correctly."""
        raise NotImplementedError

    def _save_to_state_dict(self, destination: OrderedDict, prefix: str, keep_vars: bool) -> None:
        if hasattr(self, "image_threshold"):
            destination[
                "image_threshold_class"
            ] = f"{self.image_threshold.__class__.__module__}.{self.image_threshold.__class__.__name__}"
        if hasattr(self, "pixel_threshold"):
            destination[
                "pixel_threshold_class"
            ] = f"{self.pixel_threshold.__class__.__module__}.{self.pixel_threshold.__class__.__name__}"
        if hasattr(self, "normalization_metrics"):
            normalization_class = self.normalization_metrics.__class__
            destination["normalization_class"] = f"{normalization_class.__module__}.{normalization_class.__name__}"

        return super()._save_to_state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict: OrderedDict[str, Any], strict: bool = True) -> Any:  # noqa: ANN401
        """Initialize auxiliary object."""
        if "image_threshold_class" in state_dict:
            self.image_threshold = self._get_instance(state_dict, "image_threshold_class")
        if "pixel_threshold_class" in state_dict:
            self.pixel_threshold = self._get_instance(state_dict, "pixel_threshold_class")
        if "normalization_class" in state_dict:
            self.normalization_metrics = self._get_instance(state_dict, "normalization_class")
        # Used to load metrics if there is any related data in state_dict
        self._load_metrics(state_dict)

        return super().load_state_dict(state_dict, strict)

    def _load_metrics(self, state_dict: OrderedDict[str, torch.Tensor]) -> None:
        """Load metrics from saved checkpoint."""
        self._add_metrics("pixel", state_dict)
        self._add_metrics("image", state_dict)

    def _add_metrics(self, name: str, state_dict: OrderedDict[str, torch.Tensor]) -> None:
        """Sets the pixel/image metrics.

        Args:
            name (str): is it pixel or image.
            state_dict (OrderedDict[str, Tensor]): state dict of the model.
        """
        metric_keys = [key for key in state_dict if key.startswith(f"{name}_metrics")]
        if any(metric_keys):
            if not hasattr(self, f"{name}_metrics"):
                setattr(self, f"{name}_metrics", AnomalibMetricCollection([], prefix=name))
            metrics = getattr(self, f"{name}_metrics")
            for key in metric_keys:
                class_name = key.split(".")[1]
                try:
                    metrics_module = importlib.import_module("anomalib.metrics")
                    metrics_cls = getattr(metrics_module, class_name)
                except (ImportError, AttributeError) as exception:
                    msg = f"Class {class_name} not found in module anomalib.metrics"
                    raise ImportError(msg) from exception
                logger.info("Loading %s metrics from state dict", class_name)
                metrics.add_metrics(metrics_cls())

    def _get_instance(self, state_dict: OrderedDict[str, Any], dict_key: str) -> BaseThreshold:
        """Get the threshold class from the ``state_dict``."""
        class_path = state_dict.pop(dict_key)
        module = importlib.import_module(".".join(class_path.split(".")[:-1]))
        return getattr(module, class_path.split(".")[-1])()

    @abstractproperty
    def learning_type(self) -> LearningType:
        """Learning type of the model."""
        raise NotImplementedError

    def to_torch(
        self,
        export_root: Path | str,
        transform: dict[str, Any] | AnomalibDataset | AnomalibDataModule | A.Compose,
        task: TaskType | None = None,
    ) -> Path:
        """Export AnomalibModel to torch.

        Args:
            export_root (Path): Path to the output folder.
            transform (dict[str, Any] | AnomalibDataset | AnomalibDataModule | A.Compose): Data transforms
                (augmentations) used for the model. When using ``dict``, ensure that the transform dict is in the format
                required by Albumentations.
            task (TaskType | None): Task type should be provided if transforms is of type dict or A.Compose object.
                Defaults to ``None``.

        Returns:
            Path: Path to the exported pytorch model.

        Examples:
            Assume that we have a model to train and we want to export it to torch format.

            >>> from anomalib.data import Visa
            >>> from anomalib.models import Patchcore
            >>> from anomalib.engine import Engine
            ...
            >>> datamodule = Visa()
            >>> model = Patchcore()
            >>> engine = Engine()
            ...
            >>> engine.fit(model, datamodule)

            Now that we have a model trained, we can export it to torch format.

            >>> model.to_torch(
            ...     export_root="path/to/export",
            ...     transform=datamodule.test_data.transform,
            ...     task=datamodule.test_data.task,
            ... )
        """
        export_root = _create_export_root(export_root, ExportType.TORCH)
        metadata = self.get_metadata(task=task, transform=transform)
        pt_model_path = export_root / "model.pt"
        torch.save(
            obj={"model": self.model, "metadata": metadata},
            f=pt_model_path,
        )
        return pt_model_path

    def to_onnx(
        self,
        input_size: tuple[int, int],
        export_root: Path | str,
        transform: dict[str, Any] | AnomalibDataset | AnomalibDataModule | A.Compose,
        task: TaskType | None = None,
        export_type: ExportType = ExportType.ONNX,
    ) -> Path:
        """Export model to onnx.

        Args:
            input_size (list[int] | tuple[int, int]): Image size used as the input for onnx converter.
            export_root (Path): Path to the root folder of the exported model.
            transform (dict[str, Any] | AnomalibDataset | AnomalibDataModule | A.Compose): Data transforms
                (augmentations) used for the model. When using dict, ensure that the transform dict is in the format
                required by Albumentations.
            task (TaskType | None): Task type should be provided if transforms is of type dict or A.Compose object.
                Defaults to ``None``.
            export_type (ExportType): Mode to export the model. Since this method is used by OpenVINO export as well, we
                need to pass the export type so that the right export path is created.
                Defaults to ``ExportType.ONNX``.

        Returns:
            Path: Path to the exported onnx model.

        Examples:
            Export the Lightning Model to ONNX:

            >>> from anomalib.models import Patchcore
            >>> from anomalib.data import Visa
            ...
            >>> datamodule = Visa()
            >>> model = Patchcore()
            ...
            >>> model.to_onnx(
            ...     input_size=(224, 224),
            ...     export_root="path/to/export",
            ...     transform=datamodule.test_data.transform,
            ...     task=datamodule.test_data.task
            ... )

            Using Custom Transforms:
            This example shows how to use a custom ``Compose`` object for the ``transform`` argument.

            >>> import albumentations as A
            >>> transform = A.Compose([A.Resize(224, 224), A.pytorch.ToTensorV2()])
            ...
            >>> model.to_onnx(
            ...     input_size=(224, 224),
            ...     export_root="path/to/export",
            ...     transform=transform,
            ...     task="segmentation",
            ... )
        """
        export_root = _create_export_root(export_root, export_type)
        self._write_metadata_to_json(export_root, transform, task)
        onnx_path = export_root / "model.onnx"
        torch.onnx.export(
            self.model,
            torch.zeros((1, 3, *input_size)).to(self.device),
            str(onnx_path),
            opset_version=14,
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            input_names=["input"],
            output_names=["output"],
        )

        return onnx_path

    def to_openvino(
        self,
        export_root: Path | str,
        input_size: tuple[int, int],
        transform: dict[str, Any] | AnomalibDataset | AnomalibDataModule | A.Compose,
        ov_args: dict[str, Any] | None = None,
        task: TaskType | None = None,
    ) -> Path:
        """Convert onnx model to OpenVINO IR.

        Args:
            export_root (Path): Path to the export folder.
            input_size (tuple[int, int]): Input size of the model. Used for adding metadata to the IR.
            transform (dict[str, Any] | AnomalibDataset | AnomalibDataModule | A.Compose): Data transforms
                (augmentations) used for the model. When using dict, ensure that the transform dict is in the format
                required by Albumentations.
            ov_args: Model optimizer arguments for OpenVINO model conversion.
                Defaults to ``None``.
            task (TaskType | None): Task type should be provided if transforms is of type dict or A.Compose object.
                Defaults to ``None``.

        Returns:
            Path: Path to the exported onnx model.

        Raises:
            ModuleNotFoundError: If OpenVINO is not installed.

        Returns:
            Path: Path to the exported OpenVINO IR.

        Examples:
            Export the Lightning Model to OpenVINO IR:
            This example demonstrates how to export the Lightning Model to OpenVINO IR.

            >>> from anomalib.models import Patchcore
            >>> from anomalib.data import Visa
            ...
            >>> datamodule = Visa()
            >>> model = Patchcore()
            ...
            >>> model.to_openvino(
            ...     export_root="path/to/export",
            ...     input_size=(224, 224),
            ...     transform=datamodule.test_data.transform,
            ...     task=datamodule.test_data.task
            ... )

            Using Custom Transforms:
            This example shows how to use a custom ``Compose`` object for the ``transform`` argument.

            >>> import albumentations as A
            >>> transform = A.Compose([A.Resize(224, 224), A.pytorch.ToTensorV2()])
            ...
            >>> model.to_openvino(
            ...     export_root="path/to/export",
            ...     input_size=(224, 224),
            ...     transform=transform,
            ...     task="segmentation",
            ... )

        """
        model_path = self.to_onnx(input_size, export_root, transform, task, ExportType.OPENVINO)
        ov_model_path = model_path.with_suffix(".xml")
        ov_args = {} if ov_args is None else ov_args
        if convert_model is not None and serialize is not None:
            model = convert_model(model_path, **ov_args)
            serialize(model, ov_model_path)
        else:
            logger.exception("Could not find OpenVINO methods. Please check OpenVINO installation.")
            raise ModuleNotFoundError
        return ov_model_path

    def get_metadata(
        self,
        transform: dict[str, Any] | AnomalibDataset | AnomalibDataModule | A.Compose,
        task: TaskType | None = None,
    ) -> dict[str, Any]:
        """Get metadata for the exported model.

        Args:
            transform (dict[str, Any] | AnomalibDataset | AnomalibDataModule | A.Compose): Data transforms
                (augmentations) for the model. When using dict, ensure that the transform dict is in the format
                required by Albumentations.
            task (TaskType | None): Task type should be provided if transforms is of type dict or A.Compose object.
                Defaults to None.

        Returns:
            dict[str, Any]: Metadata for the exported model.
        """
        transform = _get_transform_dict(transform)
        task = _get_task(task=task, transform=transform)

        data_metadata = {"task": task, "transform": transform}
        model_metadata = self._get_model_metadata()
        metadata = {**data_metadata, **model_metadata}

        # Convert torch tensors to python lists or values for json serialization.
        for key, value in metadata.items():
            if isinstance(value, torch.Tensor):
                metadata[key] = value.numpy().tolist()

        return metadata

    def _get_model_metadata(self) -> dict[str, torch.Tensor]:
        """Get meta data related to normalization from model.

        Returns:
            dict[str, torch.Tensor]: Model metadata
        """
        metadata = {}
        cached_metadata: dict[str, Number | torch.Tensor] = {}
        for threshold_name in ("image_threshold", "pixel_threshold"):
            if hasattr(self, threshold_name):
                cached_metadata[threshold_name] = getattr(self, threshold_name).cpu().value.item()
        if hasattr(self, "normalization_metrics") and self.normalization_metrics.state_dict() is not None:
            for key, value in self.normalization_metrics.state_dict().items():
                cached_metadata[key] = value.cpu()
        # Remove undefined values by copying in a new dict
        for key, val in cached_metadata.items():
            if not np.isinf(val).all():
                metadata[key] = val
        del cached_metadata
        return metadata

    def _write_metadata_to_json(
        self,
        export_root: Path,
        transform: dict[str, Any] | AnomalibDataset | AnomalibDataModule | A.Compose,
        task: TaskType | None = None,
    ) -> None:
        """Write metadata to json file.

        Args:
            export_root (Path): Path to the exported model.
            transform (dict[str, Any] | AnomalibDataset | AnomalibDataModule | A.Compose): Data transforms
            (augmentations) used for the model.
            model (AnomalyModule): AnomalyModule to export.
            task (TaskType | None): Task type should be provided if transforms is of type dict or A.Compose object.
                Defaults to None.
        """
        metadata = self.get_metadata(task=task, transform=transform)
        with (export_root / "metadata.json").open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, ensure_ascii=False, indent=4)


def _get_task(
    transform: dict[str, Any] | AnomalibDataset | AnomalibDataModule | A.Compose,
    task: TaskType | None = None,
) -> TaskType:
    """Get task from transform or task.

    Args:
        transform (dict[str, Any] | AnomalibDataset | AnomalibDataModule | A.Compose): If task is None, task is taken
            from transform.
        task (TaskType | None): Task type. Defaults to None.

    Raises:
        ValueError: If task is None and transform is not of type AnomalibDataset or AnomalibDataModule.

    Returns:
        TaskType: Task type.
    """
    _task = task
    if _task is None:
        if isinstance(transform, AnomalibDataset):
            _task = transform.task
        elif isinstance(transform, AnomalibDataModule):
            _task = transform.test_data.task
        else:
            logging.error(f"Task should be provided when passing transform of type {type(transform)}")
            raise ValueError
    return _task


def _get_transform_dict(
    transform_container: dict[str, Any] | AnomalibDataModule | AnomalibDataset | A.Compose,
) -> dict[str, Any]:
    """Get transform dict from transform_container.

    Args:
        transform_container (dict[str, Any] | AnomalibDataModule | AnomalibDataset | A.Compose): Transform dict
            or AnomalibDataModule or AnomalibDataset or A.Compose object. Transform is taken from container. When using
            AnomalibDataModule or AnomalibDataset, the task is also taken from the container. When passing
            transform_container as dict, ensure that the transform dict is in the format required by Albumentations.

    Raises:
        KeyError: If transform_container is dict and does not contain the required keys.
        TypeError: If transform_container is not dict, AnomalibDataModule or AnomalibDataset or A.Compose object.

    Returns:
        dict[str, Any]: Transform dict.
    """
    if isinstance(transform_container, dict):
        try:
            A.from_dict(transform_container)
            transform = transform_container
        except KeyError as exception:
            logging.exception(
                f"Unsupported transform: {transform_container}."
                " Ensure that the transform dict is in the format required by Albumentations.",
            )
            raise KeyError from exception
    elif isinstance(transform_container, A.Compose):
        transform = transform_container.to_dict()
    elif isinstance(transform_container, AnomalibDataset):
        transform = transform_container.transform.to_dict()
    elif isinstance(transform_container, AnomalibDataModule):
        transform = transform_container.test_data.transform.to_dict()
    else:
        logging.error(f"Unsupported type for transform_container: {type(transform_container)}")
        raise TypeError

    return transform


def _create_export_root(export_root: str | Path, export_type: ExportType) -> Path:
    """Create export directory.

    Args:
        export_root (str | Path): Path to the root folder of the exported model.
        export_type (ExportType): Mode to export the model. Torch, ONNX or OpenVINO.

    Returns:
        Path: Path to the export directory.
    """
    export_root = Path(export_root) / "weights" / export_type.value
    export_root.mkdir(parents=True, exist_ok=True)
    return export_root
