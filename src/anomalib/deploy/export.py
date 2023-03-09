"""Utilities for optimization and OpenVINO conversion."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess  # nosec
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.types import Number

from anomalib.models.components import AnomalyModule


class ExportMode(str, Enum):
    """Model export mode."""

    ONNX = "onnx"
    OPENVINO = "openvino"


def get_model_metadata(model: AnomalyModule) -> dict[str, Tensor]:
    """Get meta data related to normalization from model.

    Args:
        model (AnomalyModule): Anomaly model which contains metadata related to normalization.

    Returns:
        dict[str, Tensor]: metadata
    """
    meta_data = {}
    cached_meta_data: dict[str, Number | Tensor] = {
        "image_threshold": model.image_threshold.cpu().value.item(),
        "pixel_threshold": model.pixel_threshold.cpu().value.item(),
    }
    if hasattr(model, "normalization_metrics") and model.normalization_metrics.state_dict() is not None:
        for key, value in model.normalization_metrics.state_dict().items():
            cached_meta_data[key] = value.cpu()
    # Remove undefined values by copying in a new dict
    for key, val in cached_meta_data.items():
        if not np.isinf(val).all():
            meta_data[key] = val
    del cached_meta_data
    return meta_data


def export(
    model: AnomalyModule,
    input_size: list[int] | tuple[int, int],
    export_mode: ExportMode,
    export_root: str | Path,
) -> None:
    """Export the model to onnx format and (optionally) convert to OpenVINO IR if export mode is set to OpenVINO.

    Metadata.json is generated regardless of export mode.

    Args:
        model (AnomalyModule): Model to convert.
        input_size (list[int] | tuple[int, int]): Image size used as the input for onnx converter.
        export_root (str | Path): Path to exported ONNX/OpenVINO IR.
        export_mode (ExportMode): Mode to export the model. ONNX or OpenVINO.
    """
    # Write metadata to json file. The file is written in the same directory as the target model.
    export_path: Path = Path(str(export_root)) / export_mode.value
    export_path.mkdir(parents=True, exist_ok=True)
    with (Path(export_path) / "meta_data.json").open("w", encoding="utf-8") as metadata_file:
        meta_data = get_model_metadata(model)
        # Convert metadata from torch
        for key, value in meta_data.items():
            if isinstance(value, Tensor):
                meta_data[key] = value.numpy().tolist()
        json.dump(meta_data, metadata_file, ensure_ascii=False, indent=4)

    onnx_path = _export_to_onnx(model, input_size, export_path)
    if export_mode == ExportMode.OPENVINO:
        _export_to_openvino(export_path, onnx_path)


def _export_to_onnx(model: AnomalyModule, input_size: list[int] | tuple[int, int], export_path: Path) -> Path:
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


def _export_to_openvino(export_path: str | Path, onnx_path: Path) -> None:
    """Convert onnx model to OpenVINO IR.

    Args:
        export_path (str | Path): Path to the root folder of the exported model.
        onnx_path (Path): Path to the exported onnx model.
    """
    optimize_command = ["mo", "--input_model", str(onnx_path), "--output_dir", str(export_path)]
    subprocess.run(optimize_command, check=True)  # nosec
