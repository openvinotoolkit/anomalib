"""Timer callback.

This module provides the `TimerCallback` for measuring training and testing time of
Anomalib models.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import time

import torch
from lightning.pytorch import Callback, LightningModule, Trainer

logger = logging.getLogger(__name__)


class TimerCallback(Callback):
    """Callback for measuring model training and testing time.

    This callback tracks the time taken for training and testing, and calculates
    throughput (frames per second) during testing.

    Examples:
        Add timer to track performance::

            from anomalib.callbacks import TimerCallback
            from anomalib.engine import Engine

            callbacks = [TimerCallback()]
            engine = Engine(callbacks=callbacks)

    Note:
        The callback automatically handles both single and multiple test dataloaders.
    """

    def __init__(self) -> None:
        self.start: float
        self.num_images: int = 0

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Call when fit begins.

        Args:
            trainer (Trainer): PyTorch Lightning trainer.
            pl_module (LightningModule): Current training module.
        """
        del trainer, pl_module  # These variables are not used.
        self.start = time.time()

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Call when fit ends.

        Args:
            trainer (Trainer): PyTorch Lightning trainer.
            pl_module (LightningModule): Current training module.
        """
        del trainer, pl_module  # Unused arguments.
        logger.info("Training took %5.2f seconds", (time.time() - self.start))

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Call when test begins.

        Args:
            trainer (Trainer): PyTorch Lightning trainer.
            pl_module (LightningModule): Current training module.

        Note:
            Sets the start time and counts total number of test images across all dataloaders.
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
        """Call when test ends.

        Args:
            trainer (Trainer): PyTorch Lightning trainer.
            pl_module (LightningModule): Current training module.

        Note:
            Calculates and logs the total testing time and throughput in FPS.
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
