"""Callback that loads model weights from the state dict."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY

from anomalib.models.components import AnomalyModule

logger = logging.getLogger(__name__)


@CALLBACK_REGISTRY
class LoadModelCallback(Callback):
    """Callback that loads the model weights from the state dict."""

    def __init__(self, weights_path):
        self.weights_path = weights_path

    # pylint: disable=unused-argument
    def setup(self, trainer: Trainer, pl_module: AnomalyModule, stage: Optional[str] = None) -> None:
        """Call when inference begins.

        Loads the model weights from ``weights_path`` into the PyTorch module.
        """
        logger.info("Loading the model from %s", self.weights_path)
        pl_module.load_state_dict(torch.load(self.weights_path, map_location=pl_module.device)["state_dict"])
