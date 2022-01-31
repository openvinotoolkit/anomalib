"""Callback that loads model weights from the state dict."""
import torch
from pytorch_lightning import Callback, LightningModule


class LoadModelCallback(Callback):
    """Callback that loads the model weights from the state dict."""

    def __init__(self, weights_path):
        self.weights_path = weights_path

    def on_test_start(self, trainer, pl_module: LightningModule) -> None:  # pylint: disable=W0613
        """Call when the test begins.

        Loads the model weights from ``weights_path`` into the PyTorch module.
        """
        pl_module.load_state_dict(torch.load(self.weights_path)["state_dict"])
