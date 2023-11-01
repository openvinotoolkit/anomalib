"""Callback that compresses a trained model by first exporting to .onnx format, and then converting to OpenVINO IR."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch import Callback

from anomalib.deploy.export import ExportMode, export_to_onnx, export_to_openvino, export_to_torch
from anomalib.models.components import AnomalyModule

logger = logging.getLogger(__name__)

# TODO(ashwinvaidya17): Drop this callback when migration to API is complete. Users should use engine.export
# CVS-123591


class ExportCallback(Callback):
    """Callback to compresses a trained model.

    Model is first exported to ``.onnx`` format, and then converted to OpenVINO IR.

    Args:
        input_size (tuple[int, int]): Tuple of image height, width
        dirpath (str): Path for model output
        filename (str): Name of output model
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        dirpath: str | Path,
        filename: str,
        export_mode: ExportMode,
    ) -> None:
        self.input_size = input_size
        self.dirpath = Path(dirpath) if isinstance(dirpath, str) else dirpath
        self.filename = filename
        self.export_mode = export_mode

    def on_train_end(self, trainer: pl.Trainer, pl_module: AnomalyModule) -> None:
        """Call when the train ends.

        Converts the model to ``onnx`` format and then calls OpenVINO's model optimizer to get the
        ``.xml`` and ``.bin`` IR files.
        """
        logger.info("Exporting the model")
        self.dirpath.mkdir(parents=True, exist_ok=True)

        if self.export_mode == ExportMode.TORCH:
            export_to_torch(
                model=pl_module,
                export_path=self.dirpath,
                transform=trainer.datamodule.test_data.transform,
                task=trainer.datamodule.test_data.task,
            )
        elif self.export_mode == ExportMode.ONNX:
            export_to_onnx(
                model=pl_module,
                input_size=self.input_size,
                export_path=self.dirpath,
                transform=trainer.datamodule.test_data.transform,
                task=trainer.datamodule.test_data.task,
            )
        else:
            export_to_openvino(
                export_path=self.dirpath,
                model=pl_module,
                input_size=self.input_size,
                transform=trainer.datamodule.test_data.transform,
                mo_args={},
                task=trainer.datamodule.test_data.task,
            )
