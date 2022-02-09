"""Model converters."""

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

from pathlib import Path
from typing import List, Union

from anomalib.deploy import export_convert
from anomalib.models import AnomalyModule


def convert_to_openvino(model: AnomalyModule, export_path: Union[Path, str], input_size: List[int]):
    """Convert the trained model to OpenVINO."""
    export_path = export_path if isinstance(export_path, Path) else Path(export_path)
    onnx_path = export_path / "model.onnx"
    export_convert(model, input_size, onnx_path, export_path)
