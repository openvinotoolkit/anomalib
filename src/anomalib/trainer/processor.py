"""Callback that attaches necessary pre/post-processing to the model."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from typing import Any

from lightning import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib import trainer
from anomalib.models import AnomalyModule

from .handlers import PostProcessor


class _ProcessorCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.post_processor = PostProcessor()

    def on_validation_batch_end(
        self,
        trainer: "trainer.AnomalibTrainer",
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if outputs is not None:
            self.post_processor.process(trainer, pl_module, outputs)

    def on_test_batch_end(
        self,
        trainer: "trainer.AnomalibTrainer",
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if outputs is not None:
            self.post_processor.process(trainer, pl_module, outputs)

    def on_predict_batch_end(
        self,
        trainer: "trainer.AnomalibTrainer",
        pl_module: AnomalyModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if outputs is not None:
            self.post_processor.process(trainer, pl_module, outputs)
