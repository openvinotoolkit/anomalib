"""Callback to measure training and testing time of a PyTorch Lightning module."""
import time

from pytorch_lightning import Callback, LightningModule


class TimerCallback(Callback):
    """Callback that measures the training and testing time of a PyTorch Lightning module."""

    # pylint: disable=unused-argument
    def __init__(self):
        self.start: float

    def on_fit_start(self, trainer, pl_module: LightningModule) -> None:
        """Set start to current time when the training starts."""
        self.start = time.time()

    def on_fit_end(self, trainer, pl_module: LightningModule) -> None:
        """Display time taken for training."""
        print(f"Training took {time.time() - self.start} seconds")

    def on_test_start(self, trainer, pl_module: LightningModule) -> None:
        """Set start to current time when the testing starts."""
        self.start = time.time()

    def on_test_end(self, trainer, pl_module: LightningModule) -> None:
        """Display time taken for testing."""
        print(f"Testing took {time.time() - self.start} seconds.")
