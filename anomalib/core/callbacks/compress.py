"""Callback that compresses a trained model by first exporting to .onnx format, and then converting to OpenVINO IR."""
import os
from typing import Tuple

import torch
from pytorch_lightning import Callback, LightningModule


class CompressModelCallback(Callback):
    """Callback to compresses a trained model.

    Model is first exported to .onnx format, and then converted to OpenVINO IR.

    Args:
        input_size (Tuple[int, int]): Tuple of image height, width
        dirpath (str): Path for model output
        filename (str): Name of output model
    """

    def __init__(self, input_size: Tuple[int, int], dirpath: str, filename: str):
        self.input_size = input_size
        self.dirpath = dirpath
        self.filename = filename

    def on_train_end(self, trainer, pl_module: LightningModule) -> None:  # pylint: disable=W0613
        """Call when the train ends."""
        os.makedirs(self.dirpath, exist_ok=True)
        onnx_path = os.path.join(self.dirpath, self.filename + ".onnx")
        height, width = self.input_size
        torch.onnx.export(
            pl_module.model,
            torch.zeros((1, 3, height, width)).to(pl_module.device),
            onnx_path,
            opset_version=11,
            input_names=["input"],
            output_names=["output"],
        )
        optimize_command = "mo --input_model " + onnx_path + " --output_dir " + self.dirpath
        os.system(optimize_command)
