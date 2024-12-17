"""Graph logging callback for model visualization.

This module provides the `GraphLogger` callback for visualizing model architectures in various logging backends.
The callback supports `TensorBoard`, `Comet`, and `Weights & Biases` (W&B).

Note:
    For W&B logging, the graph is only populated after one backward pass. This means
    it may not work for models that don't require training (e.g., `PaDiM`).
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from lightning.pytorch import Callback, LightningModule, Trainer

from anomalib.loggers import AnomalibCometLogger, AnomalibTensorBoardLogger, AnomalibWandbLogger


class GraphLogger(Callback):
    """Log model graph to respective logger.

    This callback logs the model architecture graph to the configured logger. It supports multiple
    logging backends including `TensorBoard`, `Comet`, and `Weights & Biases` (W&B).

    The callback automatically detects which logger is being used and handles the graph logging
    appropriately for each backend.

    Examples:
        Log model graph to TensorBoard::

            from anomalib.callbacks import GraphLogger
            from anomalib.loggers import AnomalibTensorBoardLogger
            from anomalib.engine import Engine

            logger = AnomalibTensorBoardLogger()
            callbacks = [GraphLogger()]
            engine = Engine(logger=logger, callbacks=callbacks)

        Log model graph to Comet::

            from anomalib.callbacks import GraphLogger
            from anomalib.loggers import AnomalibCometLogger
            from anomalib.engine import Engine

            logger = AnomalibCometLogger()
            callbacks = [GraphLogger()]
            engine = Engine(logger=logger, callbacks=callbacks)

    Notes:
        - For `TensorBoard` and `Comet`, the graph is logged at the end of training
        - For W&B, the graph is logged at the start of training but requires one backward pass
          to be populated. This means it may not work for models that don't require training
          (e.g., `PaDiM`)
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
