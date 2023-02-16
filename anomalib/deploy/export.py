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
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.types import Number

from anomalib.models.components import AnomalyModule


class ExportMode(str, Enum):
    """Model export mode."""

    ONNX = "onnx"
    OPENVINO = "openvino"


def get_metadata_from_model(model: AnomalyModule) -> dict[str, Tensor]:
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


def get_metadata_from_trainer(trainer: pl.Trainer) -> dict[str, Any]:
    """Get meta data related to normalization from PL Trainer.

    Args:
        trainer (pl.Trainer): Pytorch Lightning trainer object containing model, datamodule and the rest of the hparams.

    Returns:
        dict[str, Any]: metadata
    """
    model = trainer.model

    metadata: dict[str, Any] = {
        "task": trainer.datamodule.test_data.task,
        "transform": trainer.datamodule.test_data.transform.to_dict(),
        "image_threshold": model.image_threshold.cpu().value.item(),
        "pixel_threshold": model.pixel_threshold.cpu().value.item(),
    }

    # Add normalization metrics to the post processing metadata
    if hasattr(model, "normalization_metrics") and model.normalization_metrics.state_dict() is not None:
        for key, value in model.normalization_metrics.state_dict().items():
            metadata[key] = value.cpu()
            metadata[key] = value.cpu()

    return metadata


def export(
    trainer: pl.Trainer,
    input_size: tuple[int, int] | None = None,
    export_mode: ExportMode | None = None,
    export_root: str | Path | None = None,
) -> None:
    """Export the model to onnx format and (optionally) convert to OpenVINO IR if export mode is set to OpenVINO.

    Metadata.json is generated regardless of export mode.

    Args:
        trainer (pl.Trainer): Pytorch Lightning trainer object containing model, datamodule and the rest of the hparams.
        input_size (list[int] | tuple[int, int]| None): Image size used as the input. Defaults to None.
        export_root (str | Path| None): Path to exported ONNX/OpenVINO IR. Defaults to None.
        export_mode (ExportMode| None): Mode to export the model. ONNX or OpenVINO. Defaults to None.
    """
    # Write metadata to json file. The file is written in the same directory as the target model.

    if input_size is None:
        image_size: tuple[int, int] = trainer.model.hparams["dataset"]["image_size"]
        input_size = image_size

    if export_mode is None:
        export_mode = ExportMode(trainer.model.hparams["optimization"]["export_mode"])

    if export_root is None:
        export_root = str(trainer.default_root_dir)

    export_path = Path(export_root) / export_mode.value
    export_path.mkdir(parents=True, exist_ok=True)
    with (Path(export_path) / "meta_data.json").open("w", encoding="utf-8") as metadata_file:
        meta_data = get_metadata_from_trainer(trainer)
        # Convert metadata from torch
        for key, value in meta_data.items():
            if isinstance(value, Tensor):
                meta_data[key] = value.numpy().tolist()
        json.dump(meta_data, metadata_file, ensure_ascii=False, indent=4)

    onnx_path = _export_to_onnx(trainer.model, input_size, export_path)
    if export_mode == ExportMode.OPENVINO:
        _export_to_openvino(export_path, onnx_path)


def _export_to_onnx(model: AnomalyModule, input_size: tuple[int, int], export_path: Path) -> Path:
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
