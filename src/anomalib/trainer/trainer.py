"""Implements custom trainer for Anomalib."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from lightning import Callback
from lightning.pytorch import Trainer
from omegaconf import DictConfig

from anomalib.models import AnomalyModule
from anomalib.post_processing import NormalizationMethod
from anomalib.utils.callbacks.normalization import get_normalization_callback
from anomalib.utils.callbacks.post_processor import _PostProcessorCallback

log = logging.getLogger(__name__)


class AnomalibTrainer(Trainer):
    """Anomalib trainer.

    Note:
        Refer to PyTorch Lightning's Trainer for a list of parameters for details on other Trainer parameters.

    Args:
        callbacks: Add a callback or list of callbacks.
    """

    def __init__(
        self,
        callbacks: list[Callback] = [],
        normalizer: NormalizationMethod | DictConfig | Callback | str = NormalizationMethod.MIN_MAX,
        **kwargs,
    ) -> None:
        self.normalizer = normalizer
        super().__init__(callbacks=self._setup_callbacks(callbacks), **kwargs)

        self.lightning_module: AnomalyModule

    def _setup_callbacks(self, callbacks: list[Callback]) -> list[Callback]:
        """Setup callbacks for the trainer."""
        # Note: this needs to be changed when normalization is part of the trainer
        _callbacks: list[Callback] = [_PostProcessorCallback()]
        normalization_callback = get_normalization_callback(self.normalizer)
        if normalization_callback is not None:
            _callbacks.append(normalization_callback)
        return _callbacks + callbacks
