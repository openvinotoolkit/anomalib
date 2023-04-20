"""Fit loop."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pytorch_lightning.loops.epoch.training_epoch_loop import TrainingEpochLoop
from pytorch_lightning.loops.fit_loop import FitLoop

import anomalib.trainer as trainer  # to avoid circular import

from .validate import AnomalibValidationLoop


class AnomalibTrainingEpochLoop(TrainingEpochLoop):
    def __init__(self) -> None:
        super().__init__()
        self.trainer: trainer.AnomalibTrainer
        self.val_loop = AnomalibValidationLoop()

    def _should_check_val_epoch(self) -> bool:
        """Don't run validation before the training epoch ends.

        This is because in some models we need to compute embeddings or fit the normality model from all the collected
        training data. By default lightning starts the validation loop before ``train_epoch_end`` is called.
        """
        return False


class AnomalibFitLoop(FitLoop):
    """Fit loop for default strategy.

    One of the changes here is that ``on_train_epoch_end`` is called before the validation loop starts unlike the
    default method in lightning.
    """

    def __init__(self, min_epochs: int | None, max_epochs: int | None) -> None:
        super().__init__(min_epochs=min_epochs, max_epochs=max_epochs)
        self.trainer: trainer.AnomalibTrainer
        self.epoch_loop = AnomalibTrainingEpochLoop()

    def on_advance_end(self) -> None:
        """The idea behind this is to call ``on_train_epoch_end`` before the validation loop starts."""
        super().on_advance_end()
        self.trainer.validating = True
        self.epoch_loop._run_validation()
