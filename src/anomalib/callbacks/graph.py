"""Log model graph to respective logger."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from lightning.pytorch import Callback, LightningModule, Trainer

from anomalib.loggers import AnomalibCometLogger, AnomalibTensorBoardLogger, AnomalibWandbLogger


class GraphLogger(Callback):
    """Log model graph to respective logger.

    Examples:
        Log model graph to Tensorboard

        >>> from anomalib.callbacks import GraphLogger
        >>> from anomalib.loggers import AnomalibTensorBoardLogger
        >>> from anomalib.engine import Engine
        ...
        >>> logger = AnomalibTensorBoardLogger()
        >>> callbacks = [GraphLogger()]
        >>> engine = Engine(logger=logger, callbacks=callbacks)

        Log model graph to Comet

        >>> from anomalib.loggers import AnomalibCometLogger
        >>> from anomalib.engine import Engine
        ...
        >>> logger = AnomalibCometLogger()
        >>> callbacks = [GraphLogger()]
        >>> engine = Engine(logger=logger, callbacks=callbacks)
    """

    @staticmethod
    def on_train_start(trainer: Trainer, pl_module: LightningModule) -> None:
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

    @staticmethod
    def on_train_end(trainer: Trainer, pl_module: LightningModule) -> None:
        """Unwatch model if configured for wandb and log it model graph in Tensorboard if specified.

        Args:
            trainer: Trainer object which contans reference to loggers.
            pl_module: LightningModule object which is logged.
        """
        for logger in trainer.loggers:
            if isinstance(logger, AnomalibCometLogger | AnomalibTensorBoardLogger):
                logger.log_graph(pl_module, input_array=torch.ones((1, 3, 256, 256)))
            elif isinstance(logger, AnomalibWandbLogger):
                logger.experiment.unwatch(pl_module)
