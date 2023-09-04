"""Log model graph to respective logger."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

from anomalib.utils.loggers import (
    AnomalibCometLogger,
    AnomalibTensorBoardLogger,
    AnomalibWandbLogger,
)


class GraphLogger(Callback):
    """Log model graph to respective logger."""

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log model graph to respective logger.

        Args:
            trainer: Trainer object which contans reference to loggers.
            pl_module: LightningModule object which is logged.
        """

        for logger in trainer.loggers:
            if isinstance(logger, AnomalibWandbLogger):
                # NOTE: log graph gets populated only after one backward pass. This won't work for models which do not
                # require training such as Padim
                logger.watch(pl_module, log_graph=True, log="all")
                break

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Unwatch model if configured for wandb and log it model graph in Tensorboard if specified.

        Args:
            trainer: Trainer object which contans reference to loggers.
            pl_module: LightningModule object which is logged.
        """

        for logger in trainer.loggers:
            if isinstance(logger, (AnomalibCometLogger, AnomalibTensorBoardLogger)):
                logger.log_graph(pl_module, input_array=torch.ones((1, 3, 256, 256)))
            elif isinstance(logger, AnomalibWandbLogger):
                logger.unwatch(pl_module)  # type: ignore
