"""Base class for hooks."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from abc import ABC
from typing import List

from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT

from anomalib.models import AnomalyModule


class TrainerHooks(ABC):
    """Abstract class used to build hooks for the trainer.

    It is similar to the lightning callbacks. This makes it more explicit that the classes derived from this are meant
    for the trainer and not as a callback.

    Override any relevant hooks.
    """

    def on_run_start(self, pl_module: AnomalyModule):
        """Called at the beginning of the loop."""

    def on_run_end(self, pl_module: AnomalyModule, *args, **kwargs):
        """Called at the end of the loop."""

    def teardown(self) -> None:
        """Use to release memory etc."""

    def predict_step(self, lightning_module: AnomalyModule, outputs: List[STEP_OUTPUT]):
        """Called at the end of the predict step."""

    def train_step(self, pl_module: AnomalyModule, outputs: STEP_OUTPUT):
        """Called at the end of the train step."""

    def train_batch_end(self, pl_module: AnomalyModule, outputs: STEP_OUTPUT):
        """Called at the end of the train batch."""

    def train_epoch_end(self, pl_module: AnomalyModule, outputs: EPOCH_OUTPUT):
        """Called at the end of the train epoch."""

    def test_step(self, pl_module: AnomalyModule, outputs: STEP_OUTPUT):
        """Called at the end of the test step."""

    def test_batch_end(self, pl_module: AnomalyModule, outputs: STEP_OUTPUT):
        """Called at the end of the test batch."""

    def test_epoch_end(self, pl_module: AnomalyModule, outputs: EPOCH_OUTPUT):
        """Called at the end of the test epoch."""

    def validation_step(self, pl_module: AnomalyModule, outputs: STEP_OUTPUT):
        """Called at the end of the validation step."""

    def validation_batch_end(self, pl_module: AnomalyModule, outputs: STEP_OUTPUT):
        """Called at the end of the validation batch."""

    def validation_epoch_end(self, pl_module: AnomalyModule, outputs: EPOCH_OUTPUT):
        """Called at the end of the validation epoch."""
