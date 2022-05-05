"""Callback to measure training and testing time of a PyTorch Lightning module."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import logging
import time

from pytorch_lightning import Callback, LightningModule, Trainer

logger = logging.getLogger(__name__)


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
        logger.info("Training took %5.2f seconds", (time.time() - self.start))

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
        output = f"Testing took {testing_time} seconds\nThroughput "
        if trainer.test_dataloaders is not None:
            output += f"(batch_size={trainer.test_dataloaders[0].batch_size})"
        output += f" : {self.num_images/testing_time} FPS"
        logger.info(output)
