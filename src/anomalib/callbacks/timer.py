"""Callback to measure training and testing time of a PyTorch Lightning module."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import time

import torch
from lightning.pytorch import Callback, LightningModule, Trainer

logger = logging.getLogger(__name__)


class TimerCallback(Callback):
    """Callback that measures the training and testing time of a PyTorch Lightning module.

    Examples:
        >>> from anomalib.callbacks import TimerCallback
        >>> from anomalib.engine import Engine
        ...
        >>> callbacks = [TimerCallback()]
        >>> engine = Engine(callbacks=callbacks)
    """

    def __init__(self) -> None:
        self.start: float
        self.num_images: int = 0

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Call when fit begins.

        Sets the start time to the time training started.

        Args:
            trainer (Trainer): PyTorch Lightning trainer.
            pl_module (LightningModule): Current training module.

        Returns:
            None
        """
        del trainer, pl_module  # These variables are not used.

        self.start = time.time()

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Call when fit ends.

        Prints the time taken for training.

        Args:
            trainer (Trainer): PyTorch Lightning trainer.
            pl_module (LightningModule): Current training module.

        Returns:
            None
        """
        del trainer, pl_module  # Unused arguments.
        logger.info("Training took %5.2f seconds", (time.time() - self.start))

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Call when the test begins.

        Sets the start time to the time testing started.
        Goes over all the test dataloaders and adds the number of images in each.

        Args:
            trainer (Trainer): PyTorch Lightning trainer.
            pl_module (LightningModule): Current training module.

        Returns:
            None
        """
        del pl_module  # Unused argument.

        self.start = time.time()
        self.num_images = 0

        if trainer.test_dataloaders is not None:  # Check to placate Mypy.
            if isinstance(trainer.test_dataloaders, torch.utils.data.dataloader.DataLoader):
                self.num_images += len(trainer.test_dataloaders.dataset)
            else:
                for dataloader in trainer.test_dataloaders:
                    self.num_images += len(dataloader.dataset)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Call when the test ends.

        Prints the time taken for testing and the throughput in frames per second.

        Args:
            trainer (Trainer): PyTorch Lightning trainer.
            pl_module (LightningModule): Current training module.

        Returns:
            None
        """
        del pl_module  # Unused argument.

        testing_time = time.time() - self.start
        output = f"Testing took {testing_time} seconds\nThroughput "
        if trainer.test_dataloaders is not None:
            if isinstance(trainer.test_dataloaders, torch.utils.data.dataloader.DataLoader):
                test_data_loader = trainer.test_dataloaders
            else:
                test_data_loader = trainer.test_dataloaders[0]
            output += f"(batch_size={test_data_loader.batch_size})"
        output += f" : {self.num_images / testing_time} FPS"
        logger.info(output)
