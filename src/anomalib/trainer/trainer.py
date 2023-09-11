"""Implements custom trainer for Anomalib."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from lightning import Callback
from lightning.pytorch import Trainer

from anomalib.models import AnomalyModule

from .callbacks import PostProcessorCallback

log = logging.getLogger(__name__)


class AnomalibTrainer(Trainer):
    """Anomalib trainer.

    Note:
        Refer to PyTorch Lightning's [Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api)
         for a list of parameters for details on other Trainer parameters.

    Args:
        callbacks: Add a callback or list of callbacks.
    """

    def __init__(
        self,
        callbacks: list[Callback] = [],
        **kwargs,
    ) -> None:
        self._setup_callbacks(callbacks)
        super().__init__(callbacks=callbacks, **kwargs)

        self.lightning_module: AnomalyModule

    def _setup_callbacks(self, callbacks: list[Callback]) -> None:
        """Setup callbacks for the trainer."""
        # Note: this needs to be changed when normalization is part of the trainer
        callbacks.insert(0, PostProcessorCallback())
