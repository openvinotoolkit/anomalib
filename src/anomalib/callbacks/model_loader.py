"""Model loader callback.

This module provides the `LoadModelCallback` for loading pre-trained model weights from a state dict.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import torch
from lightning.pytorch import Callback, Trainer

from anomalib.models.components import AnomalibModule

logger = logging.getLogger(__name__)


class LoadModelCallback(Callback):
    """Callback that loads model weights from a state dict.

    This callback loads pre-trained model weights from a specified path when inference begins.

    Args:
        weights_path (str): Path to the model weights file (.pt or .pth).

    Examples:
        Load pre-trained weights::

            from anomalib.callbacks import LoadModelCallback
            from anomalib.engine import Engine

            callbacks = [LoadModelCallback(weights_path="path/to/weights.pt")]
            engine = Engine(callbacks=callbacks)
    """

    def __init__(self, weights_path: str) -> None:
        self.weights_path = weights_path

    def setup(self, trainer: Trainer, pl_module: AnomalibModule, stage: str | None = None) -> None:
        """Call when inference begins.

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance.
            pl_module (AnomalibModule): The module to load weights into.
            stage (str | None, optional): Current stage of execution. Defaults to None.

        Note:
            Loads the model weights from ``weights_path`` into the PyTorch module.
        """
        del trainer, stage  # These variables are not used.

        logger.info("Loading the model from %s", self.weights_path)
        pl_module.load_state_dict(torch.load(self.weights_path, map_location=pl_module.device)["state_dict"])
