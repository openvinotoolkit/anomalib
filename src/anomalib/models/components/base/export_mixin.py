"""Mixin for exporting models to disk."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torchmetrics import Metric
from torchvision.transforms.v2 import Transform

from anomalib import TaskType
from anomalib.data import AnomalibDataModule
from anomalib.deploy.export import CompressionType, ExportType, InferenceModel
from anomalib.metrics import create_metric_collection
from anomalib.utils.exceptions import try_import

if TYPE_CHECKING:
    from importlib.util import find_spec

    from torch.types import Number

    if find_spec("openvino") is not None:
        from openvino import CompiledModel

logger = logging.getLogger(__name__)


class ExportMixin:
    """This mixin allows exporting models to torch and ONNX/OpenVINO."""

    model: nn.Module
    transform: Transform
    configure_transforms: Callable
    device: torch.device

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
        metadata = self._get_metadata(task=task)
        pt_model_path = export_root / "model.pt"
        torch.save(
            obj={"model": inference_model, "metadata": metadata},
            f=pt_model_path,
        )
        return pt_model_path

    def to_onnx(
        self,
        export_root: Path | str,
        input_size: tuple[int, int] | None = None,
        transform: Transform | None = None,
        task: TaskType | None = None,
    ) -> Path:
        """Export model to onnx.

        Args:
            export_root (Path): Path to the root folder of the exported model.
            input_size (tuple[int, int] | None, optional): Image size used as the input for onnx converter.
                Defaults to None.
            transform (Transform, optional): Input transforms used for the model. If not provided, the transform is
                taken from the model.
                Defaults to ``None``.
            task (TaskType | None): Task type.
                Defaults to ``None``.

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
        transform = transform or self.transform or self.configure_transforms()
        inference_model = InferenceModel(model=self.model, transform=transform, disable_antialias=True)
        export_root = _create_export_root(export_root, ExportType.ONNX)
        input_shape = torch.zeros((1, 3, *input_size)) if input_size else torch.zeros((1, 3, 1, 1))
        dynamic_axes = (
            {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
            if input_size
            else {"input": {0: "batch_size", 2: "height", 3: "weight"}, "output": {0: "batch_size"}}
        )
        _write_metadata_to_json(self._get_metadata(task), export_root)
        onnx_path = export_root / "model.onnx"
        torch.onnx.export(
            inference_model,
            input_shape.to(self.device),
            str(onnx_path),
            opset_version=14,
            dynamic_axes=dynamic_axes,
            input_names=["input"],
            output_names=["output"],
        )

        return onnx_path

    def to_openvino(
        self,
        export_root: Path | str,
        input_size: tuple[int, int] | None = None,
        transform: Transform | None = None,
        compression_type: CompressionType | None = None,
        datamodule: AnomalibDataModule | None = None,
        metric: Metric | str | None = None,
        ov_args: dict[str, Any] | None = None,
        task: TaskType | None = None,
    ) -> Path:
        """Convert onnx model to OpenVINO IR.

        Args:
            export_root (Path): Path to the export folder.
            input_size (tuple[int, int] | None, optional): Input size of the model. Used for adding metadata to the IR.
                Defaults to None.
            transform (Transform, optional): Input transforms used for the model. If not provided, the transform is
                taken from the model.
                Defaults to ``None``.
            compression_type (CompressionType, optional): Compression type for better inference performance.
                Defaults to ``None``.
            datamodule (AnomalibDataModule | None, optional): Lightning datamodule.
                Must be provided if ``CompressionType.INT8_PTQ`` or ``CompressionType.INT8_ACQ`` is selected.
                Defaults to ``None``.
            metric (Metric | str | None, optional): Metric to measure quality loss when quantizing.
                Must be provided if ``CompressionType.INT8_ACQ`` is selected and must return higher value for better
                performance of the model.
                Defaults to ``None``.
            ov_args (dict | None): Model optimizer arguments for OpenVINO model conversion.
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

            Export and Quantize the Model (OpenVINO IR):
            This example demonstrates how to export and quantize the model to OpenVINO IR.

            >>> from anomalib.models import Patchcore
            >>> from anomalib.data import Visa
            >>> datamodule = Visa()
            >>> model = Patchcore()
            >>> model.to_openvino(
            ...     export_root="path/to/export",
            ...     compression_type=CompressionType.INT8_PTQ,
            ...     datamodule=datamodule,
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
        if not try_import("openvino"):
            logger.exception("Could not find OpenVINO. Please check OpenVINO installation.")
            raise ModuleNotFoundError

        import openvino as ov

        with TemporaryDirectory() as onnx_directory:
            model_path = self.to_onnx(onnx_directory, input_size, transform, task)
            export_root = _create_export_root(export_root, ExportType.OPENVINO)
            ov_model_path = export_root / "model.xml"
            ov_args = {} if ov_args is None else ov_args

            model = ov.convert_model(model_path, **ov_args)
            if compression_type and compression_type != CompressionType.FP16:
                model = self._compress_ov_model(model, compression_type, datamodule, metric, task)

            # fp16 compression is enabled by default
            compress_to_fp16 = compression_type == CompressionType.FP16
            ov.save_model(model, ov_model_path, compress_to_fp16=compress_to_fp16)
            _write_metadata_to_json(self._get_metadata(task), export_root)

        return ov_model_path

    def _compress_ov_model(
        self,
        model: "CompiledModel",
        compression_type: CompressionType | None = None,
        datamodule: AnomalibDataModule | None = None,
        metric: Metric | str | None = None,
        task: TaskType | None = None,
    ) -> "CompiledModel":
        """Compress OpenVINO model with NNCF.

            model (CompiledModel): Model already exported to OpenVINO format.
            compression_type (CompressionType, optional): Compression type for better inference performance.
                Defaults to ``None``.
            datamodule (AnomalibDataModule | None, optional): Lightning datamodule.
                Must be provided if ``CompressionType.INT8_PTQ`` or ``CompressionType.INT8_ACQ`` is selected.
                Defaults to ``None``.
            metric (Metric | str | None, optional): Metric to measure quality loss when quantizing.
                Must be provided if ``CompressionType.INT8_ACQ`` is selected and must return higher value for better
                performance of the model.
                Defaults to ``None``.
            task (TaskType | None): Task type.
                Defaults to ``None``.

        Returns:
            model (CompiledModel): Model in the OpenVINO format compressed with NNCF quantization.
        """
        if not try_import("nncf"):
            logger.exception("Could not find NCCF. Please check NNCF installation.")
            raise ModuleNotFoundError

        import nncf

        if compression_type == CompressionType.INT8:
            model = nncf.compress_weights(model)
        elif compression_type == CompressionType.INT8_PTQ:
            model = self._post_training_quantization_ov(model, datamodule)
        elif compression_type == CompressionType.INT8_ACQ:
            model = self._accuracy_control_quantization_ov(model, datamodule, metric, task)
        else:
            msg = f"Unrecognized compression type: {compression_type}"
            raise ValueError(msg)

        return model

    @staticmethod
    def _post_training_quantization_ov(
        model: "CompiledModel",
        datamodule: AnomalibDataModule | None = None,
    ) -> "CompiledModel":
        """Post-Training Quantization model with NNCF.

            model (CompiledModel): Model already exported to OpenVINO format.
            datamodule (AnomalibDataModule | None, optional): Lightning datamodule.
                Must be provided if ``CompressionType.INT8_PTQ`` or ``CompressionType.INT8_ACQ`` is selected.
                Defaults to ``None``.

        Returns:
            model (CompiledModel): Quantized model.
        """
        import nncf

        if datamodule is None:
            msg = "Datamodule must be provided for OpenVINO INT8_PTQ compression"
            raise ValueError(msg)
        datamodule.setup("fit")

        model_input = model.input(0)

        if model_input.partial_shape[0].is_static:
            datamodule.train_batch_size = model_input.shape[0]

        dataloader = datamodule.val_dataloader()
        if len(dataloader.dataset) < 300:
            logger.warning(
                f">300 images recommended for INT8 quantization, found only {len(dataloader.dataset)} images",
            )

        calibration_dataset = nncf.Dataset(dataloader, lambda x: x["image"])
        return nncf.quantize(model, calibration_dataset)

    @staticmethod
    def _accuracy_control_quantization_ov(
        model: "CompiledModel",
        datamodule: AnomalibDataModule | None = None,
        metric: Metric | str | None = None,
        task: TaskType | None = None,
    ) -> "CompiledModel":
        """Accuracy-Control Quantization with NNCF.

            model (CompiledModel): Model already exported to OpenVINO format.
            datamodule (AnomalibDataModule | None, optional): Lightning datamodule.
                Must be provided if ``CompressionType.INT8_PTQ`` or ``CompressionType.INT8_ACQ`` is selected.
                Defaults to ``None``.
            metric (Metric | str | None, optional): Metric to measure quality loss when quantizing.
                Must be provided if ``CompressionType.INT8_ACQ`` is selected and must return higher value for better
                performance of the model.
                Defaults to ``None``.
            task (TaskType | None): Task type.
                Defaults to ``None``.

        Returns:
            model (CompiledModel): Quantized model.
        """
        import nncf

        if datamodule is None:
            msg = "Datamodule must be provided for OpenVINO INT8_PTQ compression"
            raise ValueError(msg)
        datamodule.setup("fit")

        if metric is None:
            msg = "Metric must be provided for OpenVINO INT8_ACQ compression"
            raise ValueError(msg)

        model_input = model.input(0)

        if model_input.partial_shape[0].is_static:
            datamodule.train_batch_size = model_input.shape[0]
            datamodule.eval_batch_size = model_input.shape[0]

        dataloader = datamodule.val_dataloader()
        if len(dataloader.dataset) < 300:
            logger.warning(
                f">300 images recommended for INT8 quantization, found only {len(dataloader.dataset)} images",
            )

        calibration_dataset = nncf.Dataset(dataloader, lambda x: x["image"])
        validation_dataset = nncf.Dataset(datamodule.test_dataloader())

        if isinstance(metric, str):
            metric = create_metric_collection([metric])[metric]

        # validation function to evaluate the quality loss after quantization
        def val_fn(nncf_model: "CompiledModel", validation_data: Iterable) -> float:
            for batch in validation_data:
                preds = torch.from_numpy(nncf_model(batch["image"])[0])
                target = batch["label"] if task == TaskType.CLASSIFICATION else batch["mask"][:, None, :, :]
                metric.update(preds, target)
            return metric.compute()

        return nncf.quantize_with_accuracy_control(model, calibration_dataset, validation_dataset, val_fn)

    def _get_metadata(
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
        model_metadata = {}
        cached_metadata: dict[str, Number | torch.Tensor] = {}
        for threshold_name in ("image_threshold", "pixel_threshold"):
            if hasattr(self, threshold_name):
                cached_metadata[threshold_name] = getattr(self, threshold_name).cpu().value.item()
        if hasattr(self, "normalization_metrics") and self.normalization_metrics.state_dict() is not None:
            for key, value in self.normalization_metrics.state_dict().items():
                cached_metadata[key] = value.cpu()
        # Remove undefined values by copying in a new dict
        model_metadata = {key: val for key, val in cached_metadata.items() if not np.isinf(val).all()}
        del cached_metadata
        metadata = {"task": task, **model_metadata}

        # Convert torch tensors to python lists or values for json serialization.
        for key, value in metadata.items():
            if isinstance(value, torch.Tensor):
                metadata[key] = value.numpy().tolist()

        return metadata


def _write_metadata_to_json(metadata: dict[str, Any], export_root: Path) -> None:
    """Write metadata to json file.

    Args:
        metadata (dict[str, Any]): Metadata to export.
        export_root (Path): Path to the exported model.
    """
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
