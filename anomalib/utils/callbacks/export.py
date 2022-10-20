"""Callback that compresses a trained model by first exporting to .onnx format, and then converting to OpenVINO IR."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Tuple

from pytorch_lightning import Callback
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY

from anomalib.deploy import ExportMode, export
from anomalib.models.components import AnomalyModule

logger = logging.getLogger(__name__)


@CALLBACK_REGISTRY
class ExportCallback(Callback):
    """Callback to compresses a trained model.

    Model is first exported to ``.onnx`` format, and then converted to OpenVINO IR.

    Args:
        input_size (Tuple[int, int]): Tuple of image height, width
        dirpath (str): Path for model output
        filename (str): Name of output model
    """

    def __init__(self, input_size: Tuple[int, int], dirpath: str, filename: str, export_mode: ExportMode):
        self.input_size = input_size
        self.dirpath = dirpath
        self.filename = filename
        self.export_mode = export_mode

    def on_train_end(self, trainer, pl_module: AnomalyModule) -> None:  # pylint: disable=W0613
        """Call when the train ends.

        Converts the model to ``onnx`` format and then calls OpenVINO's model optimizer to get the
        ``.xml`` and ``.bin`` IR files.
        """
        logger.info("Exporting the model")
        os.makedirs(self.dirpath, exist_ok=True)
        export(
            model=pl_module,
            input_size=self.input_size,
            export_root=self.dirpath,
            export_mode=self.export_mode,
        )
