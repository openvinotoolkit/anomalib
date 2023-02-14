"""Callback that loads model weights from the state dict."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import torch
from pytorch_lightning import Callback, Trainer

from anomalib.models.components import AnomalyModule

logger = logging.getLogger(__name__)


class LoadModelCallback(Callback):
    """Callback that loads the model weights from the state dict."""

    def __init__(self, weights_path) -> None:
        self.weights_path = weights_path

    def setup(self, trainer: Trainer, pl_module: AnomalyModule, stage: str | None = None) -> None:
        """Call when inference begins.

        Loads the model weights from ``weights_path`` into the PyTorch module.
        """
        del trainer, stage  # These variables are not used.

        logger.info("Loading the model from %s", self.weights_path)
        pl_module.load_state_dict(torch.load(self.weights_path, map_location=pl_module.device)["state_dict"])
