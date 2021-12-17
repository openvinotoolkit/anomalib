"""Callback that loads model weights from the state dict."""
import torch
from pytorch_lightning import Callback, LightningModule


class LoadModelCallback(Callback):
    """Callback that loads model weights from the state dict."""

    def __init__(self, weights_path):
        self.weights_path = weights_path

    def on_test_start(self, trainer, pl_module: LightningModule) -> None:  # pylint: disable=W0613
        """Call when the test begins."""
        pl_module.load_state_dict(torch.load(self.weights_path)["state_dict"])
