"""Callback that compresses a trained model by first exporting to .onnx format, and then converting to OpenVINO IR."""

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
from typing import Tuple, cast

from pytorch_lightning import Callback, LightningModule

from anomalib.deploy import export_convert
from anomalib.models.components import AnomalyModule


class OpenVINOCallback(Callback):
    """Callback to compresses a trained model.

    Model is first exported to ``.onnx`` format, and then converted to OpenVINO IR.

    Args:
        input_size (Tuple[int, int]): Tuple of image height, width
        dirpath (str): Path for model output
        filename (str): Name of output model
    """

    def __init__(self, input_size: Tuple[int, int], dirpath: str, filename: str):
        self.input_size = input_size
        self.dirpath = dirpath
        self.filename = filename

    def on_train_end(self, trainer, pl_module: LightningModule) -> None:  # pylint: disable=W0613
        """Call when the train ends.

        Converts the model to ``onnx`` format and then calls OpenVINO's model optimizer to get the
        ``.xml`` and ``.bin`` IR files.
        """
        os.makedirs(self.dirpath, exist_ok=True)
        onnx_path = os.path.join(self.dirpath, self.filename + ".onnx")
        pl_module = cast(AnomalyModule, pl_module)
        export_convert(
            model=pl_module,
            input_size=self.input_size,
            onnx_path=onnx_path,
            export_path=self.dirpath,
        )
