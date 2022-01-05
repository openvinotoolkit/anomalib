"""Utilities for optimization and OpenVINO conversion."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.


import os
from pathlib import Path
from typing import List, Tuple, Union

import pytorch_lightning as pl
import torch

from anomalib.core.model.anomaly_module import AnomalyModule


def export_convert(
    model: Union[pl.LightningModule, AnomalyModule],
    input_size: Union[List[int], Tuple[int, int]],
    onnx_path: Union[str, Path],
    export_path: Union[str, Path],
):
    """Export the model to onnx format and convert to OpenVINO IR.

    Args:
        model (Union[pl.LightningModule, AnomalyModule]): Model to convert.
        input_size (Union[List[int], Tuple[int, int]]): Image size used as the input for onnx converter.
        onnx_path (Union[str, Path]): Path to output onnx model.
        export_path (Union[str, Path]): Path to exported OpenVINO IR.
    """
    height, width = input_size
    torch.onnx.export(
        model,
        torch.zeros((1, 3, height, width)).to(model.device),
        onnx_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )
    optimize_command = "mo --input_model " + str(onnx_path) + " --output_dir " + str(export_path)
    os.system(optimize_command)
