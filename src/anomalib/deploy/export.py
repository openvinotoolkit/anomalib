"""Utilities for optimization and OpenVINO conversion."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess  # nosec
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.types import Number

from anomalib.data.task_type import TaskType
from anomalib.models.components import AnomalyModule


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
            export_to_openvino(export_path, onnx_path)

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
        onnx_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )

    return onnx_path


def export_to_openvino(export_path: str | Path, onnx_path: Path) -> None:
    """Convert onnx model to OpenVINO IR.

    Args:
        export_path (str | Path): Path to the root folder of the exported model.
        onnx_path (Path): Path to the exported onnx model.
    """
    optimize_command = ["mo", "--input_model", str(onnx_path), "--output_dir", str(export_path)]
    subprocess.run(optimize_command, check=True)  # nosec
