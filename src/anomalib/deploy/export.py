"""Utilities for optimization and OpenVINO conversion."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import json
import logging
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import albumentations as A  # noqa: N812
import numpy as np
import torch

from anomalib.data import AnomalibDataModule, AnomalibDataset
from anomalib.models.components import AnomalyModule
from anomalib.utils.exceptions import try_import
from anomalib.utils.types import TaskType

if TYPE_CHECKING:
    from torch.types import Number

logger = logging.getLogger("anomalib")

if try_import("openvino"):
    from openvino.runtime import serialize
    from openvino.tools.mo.convert import convert_model


class ExportMode(str, Enum):
    """Model export mode.

    Examples:
        >>> from anomalib.deploy import ExportMode
        >>> ExportMode.ONNX
        'onnx'
        >>> ExportMode.OPENVINO
        'openvino'
        >>> ExportMode.TORCH
        'torch'
    """

    ONNX = "onnx"
    OPENVINO = "openvino"
    TORCH = "torch"


def export_to_torch(
    model: AnomalyModule,
    export_path: Path | str,
    transform: dict[str, Any] | AnomalibDataset | AnomalibDataModule | A.Compose,
    task: TaskType | None = None,
) -> None:
    """Export AnomalibModel to torch.

    Args:
        model (AnomalyModule): Model to export.
        export_path (Path): Path to the output folder.
        transform (dict[str, Any] | AnomalibDataset | AnomalibDataModule | A.Compose): Data transforms (augmentations)
            used for the model. When using ``dict``, ensure that the transform dict is in the format required by
            Albumentations.
        task (TaskType | None): Task type should be provided if transforms is of type dict or A.Compose object.
            Defaults to ``None``.

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

        >>> from anomalib.deploy import export_to_torch
        ...
        >>> export_to_torch(
        ...     model=model,
        ...     export_path="path/to/export",
        ...     transform=datamodule.test_data.transform,
        ...     task=datamodule.test_data.task,
        ... )


    """
    export_path = _create_export_path(export_path, ExportMode.TORCH)
    metadata = get_metadata(task=task, transform=transform, model=model)
    torch.save(
        obj={"model": model.model, "metadata": metadata},
        f=export_path / "model.pt",
    )


def export_to_onnx(
    model: AnomalyModule,
    input_size: tuple[int, int],
    export_path: Path | str,
    transform: dict[str, Any] | AnomalibDataset | AnomalibDataModule | A.Compose,
    task: TaskType | None = None,
    export_mode: ExportMode = ExportMode.ONNX,
) -> Path:
    """Export model to onnx.

    Args:
        model (AnomalyModule): Model to export.
        input_size (list[int] | tuple[int, int]): Image size used as the input for onnx converter.
        export_path (Path): Path to the root folder of the exported model.
        transform (dict[str, Any] | AnomalibDataset | AnomalibDataModule | A.Compose): Data transforms (augmentations)
            used for the model. When using dict, ensure that the transform dict is in the format required by
            Albumentations.
        task (TaskType | None): Task type should be provided if transforms is of type dict or A.Compose object.
            Defaults to ``None``.
        export_mode (ExportMode): Mode to export the model. Since this method is used by OpenVINO export as well, we
            need to pass the export mode so that the right export path is created.
            Defaults to ``ExportMode.ONNX``.

    Returns:
        Path: Path to the exported onnx model.

    Examples:
        Export the Lightning Model to ONNX:

        >>> from anomalib.models import Patchcore
        >>> from anomalib.data import Visa
        >>> from anomalib.deploy import export_to_onnx
        ...
        >>> datamodule = Visa()
        >>> model = Patchcore()
        ...
        >>> export_to_onnx(
        ...     model=model,
        ...     input_size=(224, 224),
        ...     export_path="path/to/export",
        ...     transform=datamodule.test_data.transform,
        ...     task=datamodule.test_data.task
        ... )

        Using Custom Transforms:
        This example shows how to use a custom ``Compose`` object for the ``transform`` argument.

        >>> import albumentations as A
        >>> transform = A.Compose([A.Resize(224, 224), A.pytorch.ToTensorV2()])
        ...
        >>> export_to_onnx(
        ...     model=model,
        ...     input_size=(224, 224),
        ...     export_path="path/to/export",
        ...     transform=transform,
        ...     task="segmentation",
        ... )
    """
    export_path = _create_export_path(export_path, export_mode)
    _write_metadata_to_json(export_path, transform, model, task)
    onnx_path = export_path / "model.onnx"
    torch.onnx.export(
        model.model,
        torch.zeros((1, 3, *input_size)).to(model.device),
        str(onnx_path),
        opset_version=11,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        input_names=["input"],
        output_names=["output"],
    )

    return onnx_path


def export_to_openvino(
    export_path: Path | str,
    model: AnomalyModule,
    input_size: tuple[int, int],
    transform: dict[str, Any] | AnomalibDataset | AnomalibDataModule | A.Compose,
    mo_args: dict[str, Any] | None = None,
    task: TaskType | None = None,
) -> None:
    """Convert onnx model to OpenVINO IR.

    Args:
        export_path (Path): Path to the export folder.
        model (AnomalyModule): AnomalyModule to export.
        input_size (tuple[int, int]): Input size of the model. Used for adding metadata to the IR.
        transform (dict[str, Any] | AnomalibDataset | AnomalibDataModule | A.Compose): Data transforms (augmentations)
            used for the model. When using dict, ensure that the transform dict is in the format required by
            Albumentations.
        mo_args: Model optimizer arguments for OpenVINO model conversion.
            Defaults to ``None``.
        task (TaskType | None): Task type should be provided if transforms is of type dict or A.Compose object.
            Defaults to ``None``.

    Raises:
        ModuleNotFoundError: If OpenVINO is not installed.

    Examples:
        Export the Lightning Model to OpenVINO IR:
        This example demonstrates how to export the Lightning Model to OpenVINO IR.

        >>> from anomalib.models import Patchcore
        >>> from anomalib.data import Visa
        >>> from anomalib.deploy import export_to_openvino
        ...
        >>> datamodule = Visa()
        >>> model = Patchcore()
        ...
        >>> export_to_openvino(
        ...     export_path="path/to/export",
        ...     model=model,
        ...     input_size=(224, 224),
        ...     transform=datamodule.test_data.transform,
        ...     task=datamodule.test_data.task
        ... )

        Using Custom Transforms:
        This example shows how to use a custom ``Compose`` object for the ``transform`` argument.

        >>> import albumentations as A
        >>> transform = A.Compose([A.Resize(224, 224), A.pytorch.ToTensorV2()])
        ...
        >>> export_to_openvino(
        ...     export_path="path/to/export",
        ...     model=model,
        ...     input_size=(224, 224),
        ...     transform=transform,
        ...     task="segmentation",
        ... )

    """
    model_path = export_to_onnx(model, input_size, export_path, transform, task, ExportMode.OPENVINO)
    mo_args = {} if mo_args is None else mo_args
    if convert_model is not None and serialize is not None:
        model = convert_model(input_model=str(model_path), output_dir=str(model_path.parent), **mo_args)
        serialize(model, model_path.with_suffix(".xml"))
    else:
        logger.exception("Could not find OpenVINO methods. Please check OpenVINO installation.")
        raise ModuleNotFoundError


def get_metadata(
    model: AnomalyModule,
    transform: dict[str, Any] | AnomalibDataset | AnomalibDataModule | A.Compose,
    task: TaskType | None = None,
) -> dict[str, Any]:
    """Get metadata for the exported model.

    Args:
        model (AnomalyModule): Anomaly model which contains metadata related to normalization.
        transform (dict[str, Any] | AnomalibDataset | AnomalibDataModule | A.Compose): Data transforms (augmentations
             for the model. When using dict, ensure that the transform dict is in the format required by
             Albumentations.
        task (TaskType | None): Task type should be provided if transforms is of type dict or A.Compose object.
            Defaults to None.

    Returns:
        dict[str, Any]: Metadata for the exported model.
    """
    transform = _get_transform_dict(transform)
    task = _get_task(task=task, transform=transform)

    data_metadata = {"task": task, "transform": transform}
    model_metadata = _get_model_metadata(model)
    metadata = {**data_metadata, **model_metadata}

    # Convert torch tensors to python lists or values for json serialization.
    for key, value in metadata.items():
        if isinstance(value, torch.Tensor):
            metadata[key] = value.numpy().tolist()

    return metadata


def _get_model_metadata(model: AnomalyModule) -> dict[str, torch.Tensor]:
    """Get meta data related to normalization from model.

    Args:
        model (AnomalyModule): Anomaly model which contains metadata related to normalization.

    Returns:
        dict[str, torch.Tensor]: Model metadata
    """
    metadata = {}
    cached_metadata: dict[str, Number | torch.Tensor] = {}
    for threshold_name in ("image_threshold", "pixel_threshold"):
        if hasattr(model, threshold_name):
            cached_metadata[threshold_name] = getattr(model, threshold_name).cpu().value.item()
    if hasattr(model, "normalization_metrics") and model.normalization_metrics.state_dict() is not None:
        for key, value in model.normalization_metrics.state_dict().items():
            cached_metadata[key] = value.cpu()
    # Remove undefined values by copying in a new dict
    for key, val in cached_metadata.items():
        if not np.isinf(val).all():
            metadata[key] = val
    del cached_metadata
    return metadata


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


def _create_export_path(export_root: str | Path, export_mode: ExportMode) -> Path:
    """Create export directory.

    Args:
        export_root (str | Path): Path to the root folder of the exported model.
        export_mode (ExportMode): Mode to export the model. Torch, ONNX or OpenVINO.

    Returns:
        Path: Path to the export directory.
    """
    export_path = Path(export_root) / "weights" / export_mode.value
    export_path.mkdir(parents=True, exist_ok=True)
    return export_path


def _write_metadata_to_json(
    export_path: Path,
    transform: dict[str, Any] | AnomalibDataset | AnomalibDataModule | A.Compose,
    model: AnomalyModule,
    task: TaskType | None = None,
) -> None:
    """Write metadata to json file.

    Args:
        export_path (Path): Path to the exported model.
        transform (dict[str, Any] | AnomalibDataset | AnomalibDataModule | A.Compose): Data transforms (augmentations)
            used for the model.
        model (AnomalyModule): AnomalyModule to export.
        task (TaskType | None): Task type should be provided if transforms is of type dict or A.Compose object.
            Defaults to None.
    """
    metadata = get_metadata(task=task, transform=transform, model=model)
    with (export_path / "metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=4)
