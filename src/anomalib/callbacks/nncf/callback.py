"""NNCF optimization callback.

This module provides the `NNCFCallback` for optimizing neural networks using Intel's Neural Network
Compression Framework (NNCF). The callback handles model compression techniques like quantization
and pruning.

Note:
    The callback assumes that the Lightning module contains a 'model' attribute which is the
    PyTorch module to be compressed.
"""

# Copyright (C) 2022-2025 Intel Corporation
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
    """Callback for NNCF model compression.

    This callback handles the compression of PyTorch models using NNCF during training.
    It supports various compression techniques like quantization and pruning.

    Args:
        config (dict): NNCF configuration dictionary that specifies the compression
            parameters and algorithms to be applied. See the NNCF documentation for
            details on configuration options.
        export_dir (str | None, optional): Directory path where the exported models will be saved.
            If provided, the following files will be exported:

            - ONNX model file (`model_nncf.onnx`)
            - OpenVINO IR files (`model_nncf.xml` and `model_nncf.bin`)

            If ``None``, model export will be skipped. Defaults to ``None``.

    Examples:
        Configure NNCF quantization:

        >>> nncf_config = {
        ...     "input_info": {"sample_size": [1, 3, 224, 224]},
        ...     "compression": {"algorithm": "quantization"}
        ... }
        >>> callback = NNCFCallback(config=nncf_config, export_dir="./compressed_models")
        >>> trainer = pl.Trainer(callbacks=[callback])

    Note:
        - The callback assumes that the Lightning module contains a ``model`` attribute which is the
          PyTorch module to be compressed.
        - The compression is initialized using the validation dataloader since it contains both normal
          and anomalous samples, unlike the training set which only has normal samples.
        - Model export requires OpenVINO's Model Optimizer (``mo``) to be available in the system PATH.

    See Also:
        - :class:`lightning.pytorch.Callback`: Base callback class
        - :class:`nncf.NNCFConfig`: NNCF configuration class
        - :func:`nncf.torch.register_default_init_args`: Register initialization arguments
        - :func:`anomalib.callbacks.nncf.utils.wrap_nncf_model`: Wrap model for NNCF compression
    """

    def __init__(self, config: dict, export_dir: str | None = None) -> None:
        self.export_dir = export_dir
        self.config = NNCFConfig(config)
        self.nncf_ctrl: CompressionAlgorithmController | None = None

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str | None = None) -> None:
        """Initialize NNCF compression when training begins.

        This method is called when training or testing begins. It wraps the PyTorch model
        using the NNCF compression controller to prepare it for compression during training.

        Args:
            trainer (pl.Trainer): PyTorch Lightning trainer instance
            pl_module (pl.LightningModule): The Lightning module containing the model to compress
            stage (str | None, optional): Current stage of training. Defaults to ``None``.
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
        """Prepare compression before each training batch.

        Called at the beginning of each training batch to update the compression
        scheduler for the next step.

        Args:
            trainer (pl.Trainer): PyTorch Lightning trainer instance
            pl_module (pl.LightningModule): The Lightning module being trained
            batch (Any): Current batch of data
            batch_idx (int): Index of current batch
            unused (int, optional): Unused parameter. Defaults to ``0``.
        """
        del trainer, pl_module, batch, batch_idx, unused  # These variables are not used.

        if self.nncf_ctrl:
            self.nncf_ctrl.scheduler.step()

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Prepare compression before each training epoch.

        Called at the beginning of each training epoch to update the compression
        scheduler for the next epoch.

        Args:
            trainer (pl.Trainer): PyTorch Lightning trainer instance
            pl_module (pl.LightningModule): The Lightning module being trained
        """
        del trainer, pl_module  # `trainer` and `pl_module` variables are not used.

        if self.nncf_ctrl:
            self.nncf_ctrl.scheduler.epoch_step()

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Export the compressed model when training ends.

        This method handles the export of the compressed model to ONNX format and
        optionally converts it to OpenVINO IR format if the export directory is specified.

        Args:
            trainer (pl.Trainer): PyTorch Lightning trainer instance
            pl_module (pl.LightningModule): The trained Lightning module

        Note:
            - Requires OpenVINO's Model Optimizer (``mo``) to be available in the system PATH
            - Creates the export directory if it doesn't exist
            - Exports ONNX model as ``model_nncf.onnx``
            - Converts ONNX to OpenVINO IR format using ``mo``
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
