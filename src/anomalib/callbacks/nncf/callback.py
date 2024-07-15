"""Callbacks for NNCF optimization."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import subprocess  # nosec B404
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightning.pytorch as pl
from lightning.pytorch import Callback
from nncf import NNCFConfig
from nncf.torch import register_default_init_args

from anomalib.callbacks.nncf.utils import InitLoader, wrap_nncf_model

if TYPE_CHECKING:
    from nncf.api.compression import CompressionAlgorithmController


class NNCFCallback(Callback):
    """Callback for NNCF compression.

    Assumes that the pl module contains a 'model' attribute, which is
    the PyTorch module that must be compressed.

    Args:
        config (dict): NNCF Configuration
        export_dir (Str): Path where the export `onnx` and the OpenVINO `xml` and `bin` IR are saved.
                          If None model will not be exported.
    """

    def __init__(self, config: dict, export_dir: str | None = None) -> None:
        self.export_dir = export_dir
        self.config = NNCFConfig(config)
        self.nncf_ctrl: CompressionAlgorithmController | None = None

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str | None = None) -> None:
        """Call when fit or test begins.

        Takes the pytorch model and wraps it using the compression controller
        so that it is ready for nncf fine-tuning.
        """
        del stage  # `stage` variable is not used.

        if self.nncf_ctrl is not None:
            return

        # Get validate subset to initialize quantization,
        # because train subset does not contain anomalous images.
        init_loader = InitLoader(trainer.datamodule.val_dataloader())
        config = register_default_init_args(self.config, init_loader)

        self.nncf_ctrl, pl_module.model = wrap_nncf_model(
            model=pl_module.model,
            config=config,
            dataloader=trainer.datamodule.train_dataloader(),
            init_state_dict=None,  # type: ignore[arg-type]
        )

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        """Call when the train batch begins.

        Prepare compression method to continue training the model in the next step.
        """
        del trainer, pl_module, batch, batch_idx, unused  # These variables are not used.

        if self.nncf_ctrl:
            self.nncf_ctrl.scheduler.step()

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Call when the train epoch starts.

        Prepare compression method to continue training the model in the next epoch.
        """
        del trainer, pl_module  # `trainer` and `pl_module` variables are not used.

        if self.nncf_ctrl:
            self.nncf_ctrl.scheduler.epoch_step()

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Call when the train ends.

        Exports onnx model and if compression controller is not None, uses the onnx model to generate the OpenVINO IR.
        """
        del trainer, pl_module  # `trainer` and `pl_module` variables are not used.

        if self.export_dir is None or self.nncf_ctrl is None:
            return

        Path(self.export_dir).mkdir(parents=True, exist_ok=True)
        onnx_path = str(Path(self.export_dir) / "model_nncf.onnx")
        self.nncf_ctrl.export_model(onnx_path)

        optimize_command = ["mo", "--input_model", onnx_path, "--output_dir", self.export_dir]
        # TODO(samet-akcay): Check if mo can be done via python API
        # CVS-122665
        subprocess.run(optimize_command, check=True)  # noqa: S603  # nosec B603
