"""Callback to measure training and testing time of a PyTorch Lightning module"""
import time

from pytorch_lightning import Callback, LightningModule


class TimerCallback(Callback):
    """Callback that measures the training and testing time of a PyTorch Lightning module."""

    def __init__(self):
        self.start = None

    def on_fit_start(self, trainer, pl_module: LightningModule) -> None:  # pylint: disable=W0613
        """Called when fit begins"""
        self.start = time.time()

    def on_fit_end(self, trainer, pl_module: LightningModule) -> None:  # pylint: disable=W0613
        """Called when fit ends"""
        print(f"Training took {time.time() - self.start} seconds")

    def on_test_start(self, trainer, pl_module: LightningModule) -> None:  # pylint: disable=W0613
        """Called when the test begins."""
        self.start = time.time()

    def on_test_end(self, trainer, pl_module: LightningModule) -> None:  # pylint: disable=W0613
        """Called when the test ends."""
        print(f"Testing took {time.time() - self.start} seconds.")
