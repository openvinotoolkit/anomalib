"""Model converters."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import List, Union

from anomalib.deploy import export_convert
from anomalib.models import AnomalyModule


def convert_to_openvino(model: AnomalyModule, export_path: Union[Path, str], input_size: List[int]):
    """Convert the trained model to OpenVINO."""
    export_path = export_path if isinstance(export_path, Path) else Path(export_path)
    onnx_path = export_path / "model.onnx"
    export_convert(model, input_size, onnx_path, export_path)
