"""Graph logging callback for model visualization.

This module provides the :class:`GraphLogger` callback for visualizing model architectures in various logging backends.
The callback supports TensorBoard, Comet, and Weights & Biases (W&B) logging.

The callback automatically detects which logger is being used and
handles the graph logging appropriately for each backend.

Example:
    Log model graph to TensorBoard:

    >>> from anomalib.callbacks import GraphLogger
    >>> from anomalib.loggers import AnomalibTensorBoardLogger
    >>> from anomalib.engine import Engine
    >>> logger = AnomalibTensorBoardLogger()
    >>> callbacks = [GraphLogger()]
    >>> engine = Engine(logger=logger, callbacks=callbacks)

    Log model graph to Comet:

    >>> from anomalib.callbacks import GraphLogger
    >>> from anomalib.loggers import AnomalibCometLogger
    >>> from anomalib.engine import Engine
    >>> logger = AnomalibCometLogger()
    >>> callbacks = [GraphLogger()]
    >>> engine = Engine(logger=logger, callbacks=callbacks)

Note:
    For TensorBoard and Comet, the graph is logged at the end of training.
    For W&B, the graph is logged at the start of training but requires one backward pass
    to be populated. This means it may not work for models that don't require training
    (e.g., :class:`PaDiM`).
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from lightning.pytorch import Callback, LightningModule, Trainer

from anomalib.loggers import AnomalibCometLogger, AnomalibTensorBoardLogger, AnomalibWandbLogger


class GraphLogger(Callback):
    """Log model graph to respective logger.

    This callback logs the model architecture graph to the configured logger. It supports multiple
    logging backends including TensorBoard, Comet, and Weights & Biases (W&B).

    The callback automatically detects which logger is being used and handles the graph logging
    appropriately for each backend.

    Example:
        Create and use a graph logger:

        >>> from anomalib.callbacks import GraphLogger
        >>> from anomalib.loggers import AnomalibTensorBoardLogger
        >>> from lightning.pytorch import Trainer
        >>> logger = AnomalibTensorBoardLogger()
        >>> graph_logger = GraphLogger()
        >>> trainer = Trainer(logger=logger, callbacks=[graph_logger])

    Note:
        - For TensorBoard and Comet, the graph is logged at the end of training
        - For W&B, the graph is logged at the start of training but requires one backward pass
          to be populated. This means it may not work for models that don't require training
          (e.g., :class:`PaDiM`)
    """

    @staticmethod
    def on_train_start(trainer: Trainer, pl_module: LightningModule) -> None:
        """Log model graph to respective logger at training start.

        This method is called automatically at the start of training. For W&B logger,
        it sets up model watching with graph logging enabled.

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance containing logger references.
            pl_module (LightningModule): Lightning module instance to be logged.

        Example:
            >>> from anomalib.callbacks import GraphLogger
            >>> callback = GraphLogger()
            >>> # Called automatically by trainer
            >>> # callback.on_train_start(trainer, model)
        """
        for logger in trainer.loggers:
            if isinstance(logger, AnomalibWandbLogger):
                # NOTE: log graph gets populated only after one backward pass. This won't work for models which do not
                # require training such as Padim
                logger.watch(pl_module, log_graph=True, log="all")
                break

    @staticmethod
    def on_train_end(trainer: Trainer, pl_module: LightningModule) -> None:
        """Log model graph at training end and cleanup.

        This method is called automatically at the end of training. It:
        - Logs the model graph for TensorBoard and Comet loggers
        - Unwatches the model for W&B logger

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance containing logger references.
            pl_module (LightningModule): Lightning module instance to be logged.

        Example:
            >>> from anomalib.callbacks import GraphLogger
            >>> callback = GraphLogger()
            >>> # Called automatically by trainer
            >>> # callback.on_train_end(trainer, model)
        """
        for logger in trainer.loggers:
            if isinstance(logger, AnomalibCometLogger | AnomalibTensorBoardLogger):
                logger.log_graph(pl_module, input_array=torch.ones((1, 3, 256, 256)))
            elif isinstance(logger, AnomalibWandbLogger):
                logger.experiment.unwatch(pl_module)
