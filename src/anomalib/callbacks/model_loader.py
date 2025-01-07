"""Model loader callback.

This module provides the :class:`LoadModelCallback` for loading pre-trained model weights from a state dict.

The callback loads model weights from a specified path when inference begins. This is useful for loading
pre-trained models for inference or fine-tuning.

Example:
    Load pre-trained weights and create a trainer:

    >>> from anomalib.callbacks import LoadModelCallback
    >>> from anomalib.engine import Engine
    >>> from anomalib.models import Padim
    >>> model = Padim()
    >>> callbacks = [LoadModelCallback(weights_path="path/to/weights.pt")]
    >>> engine = Engine(model=model, callbacks=callbacks)

Note:
    The weights file should be a PyTorch state dict saved with either a ``.pt`` or ``.pth`` extension.
    The state dict should contain a ``"state_dict"`` key with the model weights.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import torch
from lightning.pytorch import Callback, Trainer

from anomalib.models.components import AnomalibModule

logger = logging.getLogger(__name__)


class LoadModelCallback(Callback):
    """Callback that loads model weights from a state dict.

    This callback loads pre-trained model weights from a specified path when inference begins.
    The weights are loaded into the model's state dict using the device specified by the model.

    Args:
        weights_path (str): Path to the model weights file (``.pt`` or ``.pth``).
            The file should contain a state dict with a ``"state_dict"`` key.

    Examples:
        Create a callback and use it with a trainer:

        >>> from anomalib.callbacks import LoadModelCallback
        >>> from anomalib.engine import Engine
        >>> from anomalib.models import Padim
        >>> model = Padim()
        >>> # Create callback with path to weights
        >>> callback = LoadModelCallback(weights_path="path/to/weights.pt")
        >>> # Use callback with engine
        >>> engine = Engine(model=model, callbacks=[callback])

    Note:
        The callback automatically handles device mapping when loading weights.
    """

    def __init__(self, weights_path: str) -> None:
        self.weights_path = weights_path

    def setup(self, trainer: Trainer, pl_module: AnomalibModule, stage: str | None = None) -> None:
        """Call when inference begins.

        This method is called by PyTorch Lightning when inference begins. It loads the model
        weights from the specified path into the module's state dict.

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance.
            pl_module (AnomalibModule): The module to load weights into.
            stage (str | None, optional): Current stage of execution. Defaults to ``None``.

        Note:
            The weights are loaded using ``torch.load`` with automatic device mapping based on
            the module's device. The state dict is expected to have a ``"state_dict"`` key
            containing the model weights.
        """
        del trainer, stage  # These variables are not used.

        logger.info("Loading the model from %s", self.weights_path)
        pl_module.load_state_dict(torch.load(self.weights_path, map_location=pl_module.device)["state_dict"])
