"""NNCF Callback."""

import os
from typing import Any, Dict, Iterator, Optional, Tuple

import pytorch_lightning as pl
import yaml
from nncf import NNCFConfig
from nncf.api.compression import CompressionAlgorithmController, CompressionScheduler
from nncf.torch import create_compressed_model, register_default_init_args
from nncf.torch.initialization import PTInitializingDataLoader
from omegaconf import OmegaConf
from pytorch_lightning import Callback
from torch.utils.data.dataloader import DataLoader

# pylint: disable=signature-differs,arguments-differ


def criterion_fn(outputs, criterion):
    """Calls the criterion function on outputs."""
    return criterion(outputs)


class InitLoader(PTInitializingDataLoader):
    """Initializing data loader for NNCF to be used with unsupervised training algorithms."""

    def __init__(self, data_loader: DataLoader):
        super().__init__(data_loader)
        self._data_loader_iter: Iterator

    def __iter__(self):
        """Create iterator for dataloader."""
        self._data_loader_iter = iter(self._data_loader)
        return self

    def __next__(self) -> Any:
        """Return next item from dataloader iterator."""
        loaded_item = next(self._data_loader_iter)
        return loaded_item["image"]

    def get_inputs(self, dataloader_output) -> Tuple[Tuple, Dict]:
        """Get input to model.

        Returns:
            (dataloader_output,), {}: Tuple[Tuple, Dict]: The current model call to be made during
            the initialization process
        """
        return (dataloader_output,), {}

    def get_target(self, _):
        """Return structure for ground truth in loss criterion based on dataloader output.

        Returns:
            None
        """
        return None


class NNCFCallback(Callback):
    """Callback for NNCF compression.

    Assumes that the pl module contains a 'model' attribute, which is
    the PyTorch module that must be compressed.
    """

    def __init__(self, config, dirpath, filename):
        config_dict = yaml.safe_load(OmegaConf.to_yaml(config))
        self.nncf_config = NNCFConfig.from_dict(config_dict)
        self.dirpath = dirpath
        self.filename = filename

        self.comp_ctrl: Optional[CompressionAlgorithmController] = None
        self.compression_scheduler: CompressionScheduler

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, _stage: Optional[str] = None) -> None:
        """Call when fit or test begins."""
        if self.comp_ctrl is None:
            # NOTE: trainer.datamodule returns the following error
            #   "Trainer" has no attribute "datamodule"  [attr-defined]
            init_loader = InitLoader(trainer.datamodule.train_dataloader())  # type: ignore
            nncf_config = register_default_init_args(
                self.nncf_config, init_loader, pl_module.model.loss, criterion_fn=criterion_fn
            )
            # if dump_graphs is not set to False, nncf will generate intermediate .dot files in the current dir
            self.comp_ctrl, pl_module.model = create_compressed_model(pl_module.model, nncf_config, dump_graphs=False)
            self.compression_scheduler = self.comp_ctrl.scheduler

    # NOTE: MyPy - Signature of "on_train_batch_start" incompatible with supertype "Callback"  [override]
    def on_train_batch_start(  # type: ignore
        self, trainer: pl.Trainer, _pl_module: pl.LightningModule, _batch: Any, _batch_idx: int, _dataloader_idx: int
    ) -> None:
        """Call when the train batch begins."""
        self.compression_scheduler.step()
        if self.comp_ctrl is not None:
            trainer.model.loss_val = self.comp_ctrl.loss()

    def on_train_end(self, _trainer: pl.Trainer, _pl_module: pl.LightningModule) -> None:
        """Call when the train ends."""
        os.makedirs(self.dirpath, exist_ok=True)
        onnx_path = os.path.join(self.dirpath, self.filename + ".onnx")
        if self.comp_ctrl is not None:
            self.comp_ctrl.export_model(onnx_path)
        optimize_command = "mo --input_model " + onnx_path + " --output_dir " + self.dirpath
        os.system(optimize_command)

    def on_train_epoch_end(
        self, _trainer: "pl.Trainer", _pl_module: "pl.LightningModule", _unused: Optional[Any] = None
    ) -> None:
        """Call when the train epoch ends."""
        self.compression_scheduler.epoch_step()
