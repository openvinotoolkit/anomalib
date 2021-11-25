"""Callback that compresses a trained model by first exporting to .onnx format, and then converting to OpenVINO IR."""
import os
from typing import Union

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import Callback, LightningModule


class CompressModelCallback(Callback):
    """
    Callback that compresses a trained model by first exporting to .onnx format, and then converting to OpenVINO IR.
    """

    def __init__(self, config: Union[ListConfig, DictConfig], dirpath: str, filename: str):
        self.config = config
        self.dirpath = dirpath
        self.filename = filename

    def on_train_end(self, trainer, pl_module: LightningModule) -> None:  # pylint: disable=W0613
        """Called when the train ends."""
        os.makedirs(self.dirpath, exist_ok=True)
        onnx_path = os.path.join(self.dirpath, self.filename + ".onnx")
        height, width = self.config.model.input_size
        torch.onnx.export(
            pl_module.model, torch.zeros((1, 3, height, width)).to(pl_module.device), onnx_path, opset_version=11
        )
        optimize_command = "mo --input_model " + onnx_path + " --output_dir " + self.dirpath
        os.system(optimize_command)
