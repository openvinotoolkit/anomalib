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

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize, Transform

from anomalib import LearningType, TaskType
from anomalib.data.transforms import ExportableCenterCrop
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


class InferenceModel(nn.Module):
    """Inference model for export.

    The InferenceModel is used to wrap the model and transform for exporting to torch and ONNX/OpenVINO.

    Args:
        model (nn.Module): Model to export.
        transform (Transform): Input transform for the model.
        disable_antialias (bool, optional): Disable antialiasing in the Resize transforms of the given transform. This
            is needed for ONNX/OpenVINO export, as antialiasing is not supported in the ONNX opset.
    """

    def __init__(self, model: nn.Module, transform: Transform, disable_antialias: bool = False) -> None:
        super().__init__()
        self.model = model
        self.transform = transform
        self.convert_center_crop()
        if disable_antialias:
            self.disable_antialias()

    def forward(self, batch: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Transform the input batch and pass it through the model."""
        batch = self.transform(batch)
        return self.model(batch)

    def disable_antialias(self) -> None:
        """Disable antialiasing in the Resize transforms of the given transform.

        This is needed for ONNX/OpenVINO export, as antialiasing is not supported in the ONNX opset.
        """
        if isinstance(self.transform, Resize):
            self.transform.antialias = False
        if isinstance(self.transform, Compose):
            for transform in self.transform.transforms:
                if isinstance(transform, Resize):
                    transform.antialias = False

    def convert_center_crop(self) -> None:
        """Convert CenterCrop to ExportableCenterCrop for ONNX export.

        The original CenterCrop transform is not supported in ONNX export. This method replaces the CenterCrop to
        ExportableCenterCrop, which is supported in ONNX export. For more details, see the implementation of
        ExportableCenterCrop.
        """
        if isinstance(self.transform, CenterCrop):
            self.transform = ExportableCenterCrop(size=self.transform.size)
        elif isinstance(self.transform, Compose):
            transforms = self.transform.transforms
            for index in range(len(transforms)):
                if isinstance(transforms[index], CenterCrop):
                    transforms[index] = ExportableCenterCrop(size=transforms[index].size)


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

        self._transform: Transform | None = None
        self._input_size: tuple[int, int] | None = None

        self._is_setup = False  # flag to track if setup has been called from the trainer

    @property
    def name(self) -> str:
        """Name of the model."""
        return self.__class__.__name__

    def setup(self, stage: str | None = None) -> None:
        """Calls the _setup method to build the model if the model is not already built."""
        if getattr(self, "model", None) is None or not self._is_setup:
            self._setup()
            if isinstance(stage, TrainerFn):
                # only set the flag if the stage is a TrainerFn, which means the setup has been called from a trainer
                self._is_setup = True

    def _setup(self) -> None:
        """The _setup method is used to build the torch model dynamically or adjust something about them.

        The model implementer may override this method to build the model. This is useful when the model cannot be set
        in the `__init__` method because it requires some information or data that is not available at the time of
        initialization.
        """

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
                setattr(self, f"{name}_metrics", AnomalibMetricCollection([], prefix=f"{name}_"))
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

    @property
    def transform(self) -> Transform:
        """Retrieve the model-specific transform.

        If a transform has been set using `set_transform`, it will be returned. Otherwise, we will use the
        model-specific default transform, conditioned on the input size.
        """
        return self._transform

    def set_transform(self, transform: Transform) -> None:
        """Update the transform linked to the model instance."""
        self._transform = transform

    def configure_transforms(self, image_size: tuple[int, int] | None = None) -> Transform:
        """Default transforms.

        The default transform is resize to 256x256 and normalize to ImageNet stats. Individual models can override
        this method to provide custom transforms.
        """
        logger.warning(
            "No implementation of `configure_transforms` was provided in the Lightning model. Using default "
            "transforms from the base class. This may not be suitable for your use case. Please override "
            "`configure_transforms` in your model.",
        )
        image_size = image_size or (256, 256)
        return Compose(
            [
                Resize(image_size, antialias=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        )

    @property
    def input_size(self) -> tuple[int, int] | None:
        """Return the effective input size of the model.

        The effective input size is the size of the input tensor after the transform has been applied. If the transform
        is not set, or if the transform does not change the shape of the input tensor, this method will return None.
        """
        transform = self.transform or self.configure_transforms()
        if transform is None:
            return None
        dummy_input = torch.zeros(1, 3, 1, 1)
        output_shape = transform(dummy_input).shape[-2:]
        if output_shape == (1, 1):
            return None
        return output_shape[-2:]

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Called when saving the model to a checkpoint.

        Saves the transform to the checkpoint.
        """
        checkpoint["transform"] = self.transform

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Called when loading the model from a checkpoint.

        Loads the transform from the checkpoint and calls setup to ensure that the torch model is built before loading
        the state dict.
        """
        self._transform = checkpoint["transform"]
        self.setup("load_checkpoint")

    def to_torch(
        self,
        export_root: Path | str,
        transform: Transform | None = None,
        task: TaskType | None = None,
    ) -> Path:
        """Export AnomalibModel to torch.

        Args:
            export_root (Path): Path to the output folder.
            transform (Transform, optional): Input transforms used for the model. If not provided, the transform is
                taken from the model.
                Defaults to ``None``.
            task (TaskType | None): Task type.
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
        transform = transform or self.transform or self.configure_transforms()
        inference_model = InferenceModel(model=self.model, transform=transform)
        export_root = _create_export_root(export_root, ExportType.TORCH)
        metadata = self.get_metadata(task=task)
        pt_model_path = export_root / "model.pt"
        torch.save(
            obj={"model": inference_model, "metadata": metadata},
            f=pt_model_path,
        )
        return pt_model_path

    def to_onnx(
        self,
        export_root: Path | str,
        transform: Transform | None = None,
        task: TaskType | None = None,
        export_type: ExportType = ExportType.ONNX,
    ) -> Path:
        """Export model to onnx.

        Args:
            export_root (Path): Path to the root folder of the exported model.
            transform (Transform, optional): Input transforms used for the model. If not provided, the transform is
                taken from the model.
                Defaults to ``None``.
            task (TaskType | None): Task type.
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
            ...     export_root="path/to/export",
            ...     transform=datamodule.test_data.transform,
            ...     task=datamodule.test_data.task
            ... )

            Using Custom Transforms:
            This example shows how to use a custom ``Compose`` object for the ``transform`` argument.

            >>> model.to_onnx(
            ...     export_root="path/to/export",
            ...     task="segmentation",
            ... )
        """
        # TODO(djdameln): Move export functionality to anomaly module
        # https://github.com/openvinotoolkit/anomalib/issues/1752
        transform = transform or self.transform or self.configure_transforms()
        inference_model = InferenceModel(model=self.model, transform=transform, disable_antialias=True)
        export_root = _create_export_root(export_root, export_type)
        self._write_metadata_to_json(export_root, task)
        onnx_path = export_root / "model.onnx"
        torch.onnx.export(
            inference_model,
            torch.zeros((1, 3, 1, 1)).to(self.device),
            str(onnx_path),
            opset_version=14,
            dynamic_axes={"input": {0: "batch_size", 2: "height", 3: "weight"}, "output": {0: "batch_size"}},
            input_names=["input"],
            output_names=["output"],
        )

        return onnx_path

    def to_openvino(
        self,
        export_root: Path | str,
        transform: Transform | None = None,
        ov_args: dict[str, Any] | None = None,
        task: TaskType | None = None,
    ) -> Path:
        """Convert onnx model to OpenVINO IR.

        Args:
            export_root (Path): Path to the export folder.
            transform (Transform, optional): Input transforms used for the model. If not provided, the transform is
                taken from the model.
                Defaults to ``None``.
            ov_args: Model optimizer arguments for OpenVINO model conversion.
                Defaults to ``None``.
            task (TaskType | None): Task type.
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
            ...     transform=datamodule.test_data.transform,
            ...     task=datamodule.test_data.task
            ... )

            Using Custom Transforms:
            This example shows how to use a custom ``Transform`` object for the ``transform`` argument.

            >>> from torchvision.transforms.v2 import Resize
            >>> transform = Resize(224, 224)
            ...
            >>> model.to_openvino(
            ...     export_root="path/to/export",
            ...     transform=transform,
            ...     task="segmentation",
            ... )

        """
        model_path = self.to_onnx(export_root, transform, task, ExportType.OPENVINO)
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
        task: TaskType | None = None,
    ) -> dict[str, Any]:
        """Get metadata for the exported model.

        Args:
            task (TaskType | None): Task type.
                Defaults to None.

        Returns:
            dict[str, Any]: Metadata for the exported model.
        """
        data_metadata = {"task": task}
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
        task: TaskType | None = None,
    ) -> None:
        """Write metadata to json file.

        Args:
            export_root (Path): Path to the exported model.
            transform (dict[str, Any] | AnomalibDataset | AnomalibDataModule | A.Compose): Data transforms
                (augmentations) used for the model.
            task (TaskType | None): Task type.
                Defaults to None.
        """
        metadata = self.get_metadata(task=task)
        with (export_root / "metadata.json").open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, ensure_ascii=False, indent=4)


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
