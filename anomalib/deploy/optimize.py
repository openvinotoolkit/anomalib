"""Utilities for optimization and OpenVINO conversion."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.types import Number

from anomalib.models.components import AnomalyModule


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


def export_convert(
    model: AnomalyModule,
    input_size: Union[List[int], Tuple[int, int]],
    export_mode: str,
    export_path: Optional[Union[str, Path]] = None,
):
    """Export the model to onnx format and convert to OpenVINO IR.

    Args:
        model (AnomalyModule): Model to convert.
        input_size (Union[List[int], Tuple[int, int]]): Image size used as the input for onnx converter.
        export_path (Union[str, Path]): Path to exported OpenVINO IR.
        export_mode (str): Mode to export onnx or openvino
    """
    height, width = input_size
    onnx_path = os.path.join(str(export_path), "model.onnx")
    torch.onnx.export(
        model.model,
        torch.zeros((1, 3, height, width)).to(model.device),
        onnx_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )
    if export_mode == "openvino":
        export_path = os.path.join(str(export_path), "openvino")
        optimize_command = "mo --input_model " + str(onnx_path) + " --output_dir " + str(export_path)
        os.system(optimize_command)
        with open(Path(export_path) / "meta_data.json", "w", encoding="utf-8") as metadata_file:
            meta_data = get_model_metadata(model)
            # Convert metadata from torch
            for key, value in meta_data.items():
                if isinstance(value, Tensor):
                    meta_data[key] = value.numpy().tolist()
            json.dump(meta_data, metadata_file, ensure_ascii=False, indent=4)
