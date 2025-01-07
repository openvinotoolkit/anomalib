"""Timer callback.

This module provides the :class:`TimerCallback` for measuring training and testing time of
Anomalib models. The callback tracks execution time and calculates throughput metrics.

Example:
    Add timer callback to track performance:

    >>> from anomalib.callbacks import TimerCallback
    >>> from lightning.pytorch import Trainer
    >>> callback = TimerCallback()
    >>> trainer = Trainer(callbacks=[callback])

    The callback will automatically log:
    - Total training time when training completes
    - Total testing time and throughput (FPS) when testing completes

Note:
    - The callback handles both single and multiple test dataloaders
    - Throughput is calculated as total number of images / total testing time
    - Batch size is included in throughput logging for reference
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import time

import torch
from lightning.pytorch import Callback, LightningModule, Trainer

logger = logging.getLogger(__name__)


class TimerCallback(Callback):
    """Callback for measuring model training and testing time.

    This callback tracks execution time metrics:
    - Training time: Total time taken for model training
    - Testing time: Total time taken for model testing
    - Testing throughput: Images processed per second during testing

    Example:
        Add timer to track performance:

        >>> from anomalib.callbacks import TimerCallback
        >>> from lightning.pytorch import Trainer
        >>> callback = TimerCallback()
        >>> trainer = Trainer(callbacks=[callback])

    Note:
        - The callback automatically handles both single and multiple test dataloaders
        - Throughput is calculated as: ``num_test_images / testing_time``
        - All metrics are logged using the logger specified in the trainer
    """

    def __init__(self) -> None:
        """Initialize timer callback.

        The callback initializes:
        - ``start``: Timestamp for tracking execution segments
        - ``num_images``: Counter for total test images
        """
        self.start: float
        self.num_images: int = 0

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when fit begins.

        Records the start time of the training process.

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance
            pl_module (LightningModule): The current training module

        Note:
            The trainer and module arguments are not used but kept for callback signature compatibility
        """
        del trainer, pl_module  # These variables are not used.
        self.start = time.time()

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when fit ends.

        Calculates and logs the total training time.

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance
            pl_module (LightningModule): The current training module

        Note:
            The trainer and module arguments are not used but kept for callback signature compatibility
        """
        del trainer, pl_module  # Unused arguments.
        logger.info("Training took %5.2f seconds", (time.time() - self.start))

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when test begins.

        Records test start time and counts total number of test images.

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance
            pl_module (LightningModule): The current training module

        Note:
            - Records start timestamp for testing phase
            - Counts total images across all test dataloaders if multiple are present
            - The module argument is not used but kept for callback signature compatibility
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
        """Called when test ends.

        Calculates and logs testing time and throughput metrics.

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance
            pl_module (LightningModule): The current training module

        Note:
            - Calculates total testing time
            - Computes throughput in frames per second (FPS)
            - Logs batch size along with throughput for reference
            - The module argument is not used but kept for callback signature compatibility
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
