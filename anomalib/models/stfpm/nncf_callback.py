from typing import Any, Dict
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule
import nncf
from nncf import NNCFConfig, create_compressed_model, load_state, register_default_init_args
from anomalib.models.stfpm.nncf_utils import InitLoader, criterion_fn
import torch
import time
from omegaconf import OmegaConf
import yaml
import os


class TimerCallback(Callback):

    def __init__(self):
        self.start = None

    def on_fit_start(self, trainer, pl_module: LightningModule) -> None:
        """Called when fit begins"""
        self.start = time.time()

    def on_fit_end(self, trainer, pl_module: LightningModule) -> None:
        """Called when fit ends"""
        print("Training took {} seconds".format(time.time() - self.start))

    def on_test_start(self, trainer, pl_module: LightningModule) -> None:
        """Called when the test begins."""
        self.start = time.time()

    def on_test_end(self, trainer, pl_module: LightningModule) -> None:
        """Called when the test ends."""
        print("Testing took {} seconds.".format(time.time() - self.start))


class LoadModelCallback(Callback):

    def __init__(self, weights_path):
        self.weights_path = weights_path

    def on_test_start(self, trainer, pl_module: LightningModule) -> None:
        """Called when the test begins."""
        pl_module.load_state_dict(torch.load(self.weights_path)["state_dict"])


class NNCFCallback(Callback):

    def __init__(self, config, dirpath, filename):
        config_dict = yaml.safe_load(OmegaConf.to_yaml(config))
        self.nncf_config = NNCFConfig.from_dict(config_dict)
        self.dirpath = dirpath
        self.filename = filename

    def setup(self, trainer, pl_module: LightningModule, stage: str) -> None:
        """Called when fit or test begins"""
        init_loader = InitLoader(pl_module.train_loader)
        nncf_config = register_default_init_args(self.nncf_config, init_loader, pl_module.model.loss,
                                                 criterion_fn=criterion_fn)
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
        self.comp_ctrl.export_model(os.path.join(self.dirpath, self.filename + '.onnx'))

    def on_train_epoch_end(self, trainer, pl_module: LightningModule, outputs: Any) -> None:
        self.compression_scheduler.epoch_step()
