"""
NNCF Callback
"""

import os
from typing import Any

import yaml
from nncf import NNCFConfig
from nncf.torch import create_compressed_model
from nncf.torch import register_default_init_args
from nncf.torch.initialization import PTInitializingDataLoader

from omegaconf import OmegaConf
from pytorch_lightning import Callback, LightningModule

from anomalib.datasets import get_datamodule


def criterion_fn(outputs, criterion):
    """
    criterion_fn [summary]

    Args:
        outputs ([type]): [description]
        criterion ([type]): [description]

    Returns:
        [type]: [description]
    """
    return criterion(outputs)


class InitLoader(PTInitializingDataLoader):
    """Initializing data loader for NNCF to be used with unsupervised training algorithms."""

    def __iter__(self):
        self._data_loader_iter = iter(self._data_loader)
        return self

    def __next__(self) -> Any:
        loaded_item = next(self._data_loader_iter)
        return loaded_item["image"]

    def get_inputs(self, dataloader_output):
        return (dataloader_output,), {}

    def get_target(self, dataloader_output):
        return None


class NNCFCallback(Callback):
    """Callback for NNCF compression.

    Assumes that the pl module contains a 'model' attribute, which is the PyTorch module
    that must be compressed.
    """

    def __init__(self, config, dirpath, filename):
        config_dict = yaml.safe_load(OmegaConf.to_yaml(config.optimization.nncf))
        self.nncf_config = NNCFConfig.from_dict(config_dict)
        self.dirpath = dirpath
        self.filename = filename

        # we need to create a datamodule here to obtain the init loader
        datamodule = get_datamodule(config)
        datamodule.setup()
        self.train_loader = datamodule.train_dataloader()

        self.comp_ctrl = None
        self.compression_scheduler = None

        self.mo_path = config.project.mo_path

    def setup(self, trainer, pl_module: LightningModule, stage: str) -> None:
        """Called when fit or test begins"""
        if self.comp_ctrl is None:
            init_loader = InitLoader(self.train_loader)
            nncf_config = register_default_init_args(
                self.nncf_config, init_loader, pl_module.model.loss, criterion_fn=criterion_fn
            )
            self.comp_ctrl, pl_module.model = create_compressed_model(pl_module.model, nncf_config)
            self.compression_scheduler = self.comp_ctrl.scheduler

    def on_train_batch_start(
        self, trainer, pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the train batch begins."""
        self.compression_scheduler.step()
        trainer.model.loss_val = self.comp_ctrl.loss()

    def on_train_end(self, trainer, pl_module: LightningModule) -> None:
        """Called when the train ends."""
        os.makedirs(self.dirpath, exist_ok=True)
        onnx_path = os.path.join(self.dirpath, self.filename + ".onnx")
        self.comp_ctrl.export_model(onnx_path)
        optimize_command = "python " + self.mo_path + " --input_model " + onnx_path + " --output_dir " + self.dirpath
        os.system(optimize_command)

    def on_train_epoch_end(self, trainer, pl_module: LightningModule, outputs: Any) -> None:
        """Called when the train epoch ends."""
        self.compression_scheduler.epoch_step()
