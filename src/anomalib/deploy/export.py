"""Utilities for optimization and OpenVINO conversion."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import subprocess  # nosec
from enum import Enum
from importlib.util import find_spec
from pathlib import Path
from typing import Any
from warnings import warn

import numpy as np
import torch
from torch import Tensor
from torch.types import Number

from anomalib.data.task_type import TaskType
from anomalib.models.components import AnomalyModule

logger = logging.getLogger("anomalib")

if find_spec("openvino") is not None:
    from openvino.runtime import Core, serialize
else:
    logger.warning("OpenVINO is not installed. Please install OpenVINO to use OpenVINOInferencer.")


class ExportMode(str, Enum):
    """Model export mode."""

    ONNX = "onnx"
    OPENVINO = "openvino"
    TORCH = "torch"


def get_model_metadata(model: AnomalyModule) -> dict[str, Tensor]:
    """Get meta data related to normalization from model.

    Args:
        model (AnomalyModule): Anomaly model which contains metadata related to normalization.

    Returns:
        dict[str, Tensor]: Model metadata
    """
    metadata = {}
    cached_metadata: dict[str, Number | Tensor] = {
        "image_threshold": model.image_threshold.cpu().value.item(),
        "pixel_threshold": model.pixel_threshold.cpu().value.item(),
    }
    if hasattr(model, "normalization_metrics") and model.normalization_metrics.state_dict() is not None:
        for key, value in model.normalization_metrics.state_dict().items():
            cached_metadata[key] = value.cpu()
    # Remove undefined values by copying in a new dict
    for key, val in cached_metadata.items():
        if not np.isinf(val).all():
            metadata[key] = val
    del cached_metadata
    return metadata


def get_metadata(task: TaskType, transform: dict[str, Any], model: AnomalyModule) -> dict[str, Any]:
    """Get metadata for the exported model.

    Args:
        task (TaskType): Task type.
        transform (dict[str, Any]): Transform used for the model.
        model (AnomalyModule): Model to export.
        export_mode (ExportMode): Mode to export the model. Torch, ONNX or OpenVINO.

    Returns:
        dict[str, Any]: Metadata for the exported model.
    """
    data_metadata = {"task": task, "transform": transform}
    model_metadata = get_model_metadata(model)
    metadata = {**data_metadata, **model_metadata}

    # Convert torch tensors to python lists or values for json serialization.
    for key, value in metadata.items():
        if isinstance(value, Tensor):
            metadata[key] = value.numpy().tolist()

    return metadata


def export(
    task: TaskType,
    transform: dict[str, Any],
    input_size: tuple[int, int],
    model: AnomalyModule,
    export_mode: ExportMode,
    export_root: str | Path,
) -> None:
    """Export the model to onnx format and (optionally) convert to OpenVINO IR if export mode is set to OpenVINO.

    Args:
        task (TaskType): Task type.
        transform (dict[str, Any]): Data transforms (augmentatiions) used for the model.
        input_size (tuple[int, int]): Input size of the model.
        model (AnomalyModule): Anomaly model to export.
        export_mode (ExportMode): Mode to export the model. Torch, ONNX or OpenVINO.
        export_root (str | Path): Path to exported Torch, ONNX or OpenVINO IR.
    """
    # Create export directory.
    export_path = Path(export_root) / "weights" / export_mode.value
    export_path.mkdir(parents=True, exist_ok=True)

    # Get metadata.
    metadata = get_metadata(task, transform, model)

    if export_mode == ExportMode.TORCH:
        export_to_torch(model, metadata, export_path)

    elif export_mode in (ExportMode.ONNX, ExportMode.OPENVINO):
        # Write metadata to json file. The file is written in the same directory as the target model.
        with (Path(export_path) / "metadata.json").open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, ensure_ascii=False, indent=4)

        # Export model to onnx and convert to OpenVINO IR if export mode is set to OpenVINO.
        onnx_path = export_to_onnx(model, input_size, export_path)
        if export_mode == ExportMode.OPENVINO:
            export_to_openvino(export_path, onnx_path, metadata, input_size)

    else:
        raise ValueError(f"Unknown export mode {export_mode}")


def export_to_torch(model: AnomalyModule, metadata: dict[str, Any], export_path: Path) -> None:
    """Export AnomalibModel to torch.

    Args:
        model (AnomalyModule): Model to export.
        export_path (Path): Path to the folder storing the exported model.
    """
    torch.save(
        obj={"model": model.model, "metadata": metadata},
        f=export_path / "model.pt",
    )


def export_to_onnx(model: AnomalyModule, input_size: tuple[int, int], export_path: Path) -> Path:
    """Export model to onnx.

    Args:
        model (AnomalyModule): Model to export.
        input_size (list[int] | tuple[int, int]): Image size used as the input for onnx converter.
        export_path (Path): Path to the root folder of the exported model.

    Returns:
        Path: Path to the exported onnx model.
    """
    onnx_path = export_path / "model.onnx"
    torch.onnx.export(
        model.model,
        torch.zeros((1, 3, *input_size)).to(model.device),
        str(onnx_path),
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )

    return onnx_path


def export_to_openvino(
    export_path: str | Path, onnx_path: Path, metadata: dict[str, Any], input_size: tuple[int, int]
) -> None:
    """Convert onnx model to OpenVINO IR.

    Args:
        export_path (str | Path): Path to the root folder of the exported model.
        onnx_path (Path): Path to the exported onnx model.
        metadata (dict[str, Any]): Metadata for the exported model.
        input_size (tuple[int, int]): Input size of the model. Used for adding metadata to the IR.
    """
    optimize_command = ["mo", "--input_model", str(onnx_path), "--output_dir", str(export_path)]
    subprocess.run(optimize_command, check=True)  # nosec
    _add_metadata_to_ir(str(export_path) + f"/{onnx_path.with_suffix('.xml').name}", metadata, input_size)


def _add_metadata_to_ir(xml_file: str, metadata: dict[str, Any], input_size: tuple[int, int]) -> None:
    """Adds the metadata to the model IR.

    Adds the metadata to the model IR. So that it can be used with the new modelAPI.
    This is because the metadata.json is not used by the new modelAPI.
    # TODO CVS-114640
    # TODO: Remove this function when Anomalib is upgraded as the model graph will contain the required ops

    Args:
        xml_file (str): Path to the xml file.
        metadata (dict[str, Any]): Metadata to add to the model.
        input_size (tuple[int, int]): Input size of the model.
    """
    core = Core()
    model = core.read_model(xml_file)

    _metadata = {}
    for key, value in metadata.items():
        if key in ("transform", "min", "max"):
            continue
        _metadata[("model_info", key)] = value

    # Add transforms
    if "transform" in metadata:
        for transform_dict in metadata["transform"]["transform"]["transforms"]:
            transform = transform_dict["__class_fullname__"]
            if transform == "Normalize":
                _metadata[("model_info", "mean_values")] = _serialize_list([x * 255.0 for x in transform_dict["mean"]])
                _metadata[("model_info", "scale_values")] = _serialize_list([x * 255.0 for x in transform_dict["std"]])
            elif transform == "Resize":
                _metadata[("model_info", "orig_height")] = transform_dict["height"]
                _metadata[("model_info", "orig_width")] = transform_dict["width"]
            else:
                warn(f"Transform {transform} is not supported currently")

    # Since we only need the diff of max and min, we fuse the min and max into one op
    if "min" in metadata and "max" in metadata:
        _metadata[("model_info", "normalization_scale")] = metadata["max"] - metadata["min"]

    _metadata[("model_info", "reverse_input_channels")] = True
    _metadata[("model_info", "model_type")] = "AnomalyDetection"
    _metadata[("model_info", "labels")] = ["Normal", "Anomaly"]
    _metadata[("model_info", "image_shape")] = _serialize_list(input_size)

    for k, data in _metadata.items():
        model.set_rt_info(data, list(k))

    tmp_xml_path = Path(xml_file).parent / "tmp.xml"
    serialize(model, str(tmp_xml_path))
    tmp_xml_path.rename(xml_file)
    # since we create new openvino IR files, we don't need the bin file. So we delete it.
    tmp_xml_path.with_suffix(".bin").unlink()


def _serialize_list(arr: list[int] | list[float] | tuple[int, int]) -> str:
    """Serializes the list to a string.

    Args:
        arr (list[int] | list[float] | tuple[int, int]): List to serialize.

    Returns:
        str: Serialized list.
    """
    return " ".join(map(str, arr))
