from pytorch_lightning import Callback, LightningModule
import torch
import os


class CompressModelCallback(Callback):

    def __init__(self, config, dirpath, filename):
        self.config = config
        self.dirpath = dirpath
        self.filename = filename
        self.mo_path = self.config.project.mo_path

    def on_train_end(self, trainer, pl_module: LightningModule) -> None:
        """Called when the train ends."""
        onnx_path = os.path.join(self.dirpath, self.filename + '.onnx')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.onnx.export(pl_module.model, torch.zeros((1, 3, 224, 224)).to(device), onnx_path)
        optimize_command = "python " + self.mo_path + " --input_model " + onnx_path + " --output_dir " + self.dirpath
        os.system(optimize_command)
