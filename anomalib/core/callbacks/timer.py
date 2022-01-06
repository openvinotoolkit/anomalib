"""Callback to measure training and testing time of a PyTorch Lightning module."""
import time

from pytorch_lightning import Callback, LightningModule, Trainer


class TimerCallback(Callback):
    """Callback that measures the training and testing time of a PyTorch Lightning module."""

    # pylint: disable=unused-argument
    def __init__(self):
        self.start: float
        self.num_images: int = 0

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:  # pylint: disable=W0613
        """Call when fit begins.

        Sets the start time to the time training started.

        Args:
            trainer (Trainer): PyTorch Lightning trainer.
            pl_module (LightningModule): Current training module.

        Returns:
            None
        """
        self.start = time.time()

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:  # pylint: disable=W0613
        """Call when fit ends.

        Prints the time taken for training.

        Args:
            trainer (Trainer): PyTorch Lightning trainer.
            pl_module (LightningModule): Current training module.

        Returns:
            None
        """
        print(f"Training took {time.time() - self.start} seconds")

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:  # pylint: disable=W0613
        """Call when the test begins.

        Sets the start time to the time testing started.
        Goes over all the test dataloaders and adds the number of images in each.

        Args:
            trainer (Trainer): PyTorch Lightning trainer.
            pl_module (LightningModule): Current training module.

        Returns:
            None
        """
        self.start = time.time()
        self.num_images = 0

        if trainer.test_dataloaders is not None:  # Check to placate Mypy.
            for dataloader in trainer.test_dataloaders:
                self.num_images += len(dataloader.dataset)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:  # pylint: disable=W0613
        """Call when the test ends.

        Prints the time taken for testing and the throughput in frames per second.

        Args:
            trainer (Trainer): PyTorch Lightning trainer.
            pl_module (LightningModule): Current training module.

        Returns:
            None
        """
        testing_time = time.time() - self.start
        print(f"Testing took {testing_time} seconds\nThroughput: {self.num_images/testing_time} FPS")
