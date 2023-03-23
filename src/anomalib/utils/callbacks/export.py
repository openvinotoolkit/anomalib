"""Callback that compresses a trained model by first exporting to .onnx format, and then converting to OpenVINO IR."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import Callback

from anomalib.deploy import ExportMode, export
from anomalib.models.components import AnomalyModule

logger = logging.getLogger(__name__)


class ExportCallback(Callback):
    """Callback to compresses a trained model.

    Model is first exported to ``.onnx`` format, and then converted to OpenVINO IR.

    Args:
        input_size (tuple[int, int]): Tuple of image height, width
        dirpath (str): Path for model output
        filename (str): Name of output model
    """

    def __init__(self, input_size: tuple[int, int], dirpath: str, filename: str, export_mode: ExportMode) -> None:
        self.input_size = input_size
        self.dirpath = dirpath
        self.filename = filename
        self.export_mode = export_mode

    def on_train_end(self, trainer: pl.Trainer, pl_module: AnomalyModule) -> None:
        """Call when the train ends.

        Converts the model to ``onnx`` format and then calls OpenVINO's model optimizer to get the
        ``.xml`` and ``.bin`` IR files.
        """
        logger.info("Exporting the model")
        Path(self.dirpath).mkdir(parents=True, exist_ok=True)

        export(
            task=trainer.datamodule.test_data.task,
            input_size=self.input_size,
            transform=trainer.datamodule.test_data.transform.to_dict(),
            model=pl_module,
            export_root=self.dirpath,
            export_mode=self.export_mode,
        )
