"""Utilities for optimization and OpenVINO conversion."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess  # nosec
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.types import Number

from anomalib.models.components import AnomalyModule


class ExportMode(str, Enum):
    """Model export mode."""

    ONNX = "onnx"
    OPENVINO = "openvino"


def get_model_metadata(model: AnomalyModule) -> Dict[str, Tensor]:
    """Get meta data related to normalization from model.

    Args:
        model (AnomalyModule): Anomaly model which contains metadata related to normalization.

    Returns:
        Dict[str, Tensor]: metadata
    """
    meta_data = {}
    cached_meta_data: Dict[str, Union[Number, Tensor]] = {
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
    input_size: Union[List[int], Tuple[int, int]],
    output_format: ExportMode,
    export_root: Union[str, Path],
):
    """Export the model to onnx output_format and (optionally) convert to OpenVINO IR if export mode is set to OpenVINO.

    Metadata.json is generated regardless of export mode.

    Args:
        model (AnomalyModule): Model to convert.
        input_size (Union[List[int], Tuple[int, int]]): Image size used as the input for onnx converter.
        export_root (Union[str, Path]): Path to exported ONNX/OpenVINO IR.
        output_format (ExportMode): Mode to export the model. ONNX or OpenVINO.
    """
    # Write metadata to json file. The file is written in the same directory as the target model.
    export_path: Path = Path(str(export_root)) / output_format.value
    export_path.mkdir(parents=True, exist_ok=True)
    with open(Path(export_path) / "meta_data.json", "w", encoding="utf-8") as metadata_file:
        meta_data = get_model_metadata(model)
        # Convert metadata from torch
        for key, value in meta_data.items():
            if isinstance(value, Tensor):
                meta_data[key] = value.numpy().tolist()
        json.dump(meta_data, metadata_file, ensure_ascii=False, indent=4)

    onnx_path = _export_to_onnx(model, input_size, export_path)
    if output_format == ExportMode.OPENVINO:
        _export_to_openvino(export_path=export_path, input_model=onnx_path)


def _export_to_onnx(model: AnomalyModule, input_size: Union[List[int], Tuple[int, int]], export_path: Path) -> Path:
    """Export model to onnx.

    Args:
        model (AnomalyModule): Model to export.
        input_size (Union[List[int], Tuple[int, int]]): Image size used as the input for onnx converter.
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


def _export_to_openvino(export_path: Optional[Union[str, Path]], input_model: Optional[Path], **kwargs):
    """Convert onnx model to OpenVINO IR.

    Args:
        export_path (Union[str, Path]): Path to the root folder of the exported model.
        input_model (Path): Path to the exported onnx model.
        kwargs: Additional arguments to pass to the OpenVINO model optimizer. These are specific to the OpenVINO
            model optimizer.
    """
    # Get input model path
    if input_model is None:
        raise ValueError("Input model must be specified.")

    # Assign export path from parameters or use the input model's directory
    export_path = export_path if export_path is not None else Path(kwargs.pop("output_dir", input_model.parent))

    # Add model optimizer specific arguments
    optimize_command = ["mo", "--input_model", str(input_model), "--output_dir", str(export_path)]
    for key, value in kwargs.items():
        optimize_command.extend(["--" + key, str(value)])

    subprocess.run(optimize_command, check=True)  # nosec
