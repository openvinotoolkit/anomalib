import os

import torch
from omegaconf.dictconfig import DictConfig
from pytorch_lightning import Callback, LightningModule


class CompressModelCallback(Callback):
    """
    Callback that compresses a trained model by first exporting to .onnx format, and then converting to OpenVINO IR.
    """

    def __init__(self, config: DictConfig, dirpath: str, filename: str):
        self.config = config
        self.dirpath = dirpath
        self.filename = filename
        self.mo_path = self.config.project.mo_path

    def on_train_end(self, trainer, pl_module: LightningModule) -> None:
        """Called when the train ends."""
        onnx_path = os.path.join(self.dirpath, self.filename + ".onnx")
        torch.onnx.export(pl_module.model, torch.zeros((1, 3, 224, 224)).to(pl_module.device), onnx_path)
        optimize_command = "python " + self.mo_path + " --input_model " + onnx_path + " --output_dir " + self.dirpath
        os.system(optimize_command)
