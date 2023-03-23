"""Test loop."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import lru_cache
from typing import Any, List, Optional

from pytorch_lightning.loops.dataloader.evaluation_loop import EvaluationLoop
from pytorch_lightning.loops.epoch.evaluation_epoch_loop import EvaluationEpochLoop
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT

import anomalib.training as core


class AnomalibTestEpochLoop(EvaluationEpochLoop):
    def __init__(self) -> None:
        super().__init__()
        self.trainer: core.AnomalibTrainer

    def _evaluation_step(self, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        """Runs the ``test_step``."""
        outputs = super()._evaluation_step(**kwargs)
        if outputs is not None:
            self.trainer.post_processor.outputs_to_cpu(outputs)
            self.trainer.post_processor.compute_labels(outputs)
            self.trainer.post_processor.apply_thresholding(self.trainer.lightning_module, outputs)
            self.trainer.normalizer.normalize(self.trainer.lightning_module, outputs)
        return outputs

    def _evaluation_step_end(self, *args, **kwargs) -> STEP_OUTPUT | None:
        """Runs ``test_step_end`` after the end of one test step."""
        outputs = super()._evaluation_step_end(*args, **kwargs)
        # Add your code here
        return outputs

    def _on_evaluation_batch_end(self, output: Optional[STEP_OUTPUT], **kwargs: Any) -> None:
        """The ``on_test_batch_end`` hook.

        Args:
            output: The output of the performed step
            batch: The input batch for the step
            batch_idx: The index of the current batch
            dataloader_idx: Index of the dataloader producing the current batch
        """
        super()._on_evaluation_batch_end(output, **kwargs)

    @lru_cache(1)
    def _should_track_batch_outputs_for_epoch_end(self) -> bool:
        """Track batch outputs for epoch end.

        If this is not overridden, the outputs are not collected if the model does not have a ``test_step_end``
        method. This ensures that the outputs are collected even if the model does not have a ``test_step_end`` method.
        """
        return True


class AnomalibTestLoop(EvaluationLoop):
    def __init__(self) -> None:
        super().__init__()
        self.trainer: "core.AnomalibTrainer"

    def on_run_start(self, *args, **kwargs) -> None:
        """Can be used to call setup."""
        self.replace(epoch_loop=AnomalibTestEpochLoop)
        self.trainer.normalizer.set_threshold(self.trainer.lightning_module)
        return super().on_run_start(*args, **kwargs)

    def _evaluation_epoch_end(self, outputs: List[EPOCH_OUTPUT]) -> None:
        """Runs ``test_epoch_end``.

        Args:
            outputs (List[EPOCH_OUTPUT])
        """

        # with a single dataloader don't pass a 2D list | Taken from base method
        output_or_outputs: EPOCH_OUTPUT | list[EPOCH_OUTPUT] = (
            outputs[0] if len(outputs) > 0 and self.num_dataloaders == 1 else outputs
        )
        self.trainer.post_processor.update_metrics(
            self.trainer.lightning_module.image_metrics,
            self.trainer.lightning_module.pixel_metrics,
            output_or_outputs,
        )
        super()._evaluation_epoch_end(outputs)
