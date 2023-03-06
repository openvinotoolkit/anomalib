from __future__ import annotations

from functools import lru_cache
from typing import Any

from pytorch_lightning.loops.dataloader.evaluation_loop import EvaluationLoop
from pytorch_lightning.loops.epoch.evaluation_epoch_loop import EvaluationEpochLoop
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT

import anomalib.core as core


class AnomalibTestEpochLoop(EvaluationEpochLoop):
    def __init__(self) -> None:
        super().__init__()
        self.trainer: "core.AnomalibTrainer"

    def _evaluation_step_end(self, *args, **kwargs) -> STEP_OUTPUT | None:
        """Runs ``test_step`` after the end of one test step."""
        outputs = super()._evaluation_step_end(*args, **kwargs)
        self.trainer._call_custom_hooks("test_step", outputs)
        return outputs

    def _on_evaluation_batch_end(self, output: STEP_OUTPUT, **kwargs: Any) -> None:
        """The ``on_test_batch_end`` hook.

        Args:
            output: The output of the performed step
            batch: The input batch for the step
            batch_idx: The index of the current batch
            dataloader_idx: Index of the dataloader producing the current batch
        """
        super()._on_evaluation_batch_end(output, **kwargs)
        self.trainer._call_custom_hooks("test_batch_end", output)

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
        self.replace(epoch_loop=AnomalibTestEpochLoop)
        return super().on_run_start(*args, **kwargs)

    def _evaluation_epoch_end(self, outputs: list[EPOCH_OUTPUT]):
        """Runs ``test_epoch_end``

        Adds on top of methods copied from the base class.
        """
        super()._evaluation_epoch_end(outputs)

        # with a single dataloader don't pass a 2D list | Taken from base method
        output_or_outputs: EPOCH_OUTPUT | list[EPOCH_OUTPUT] = (
            outputs[0] if len(outputs) > 0 and self.num_dataloaders == 1 else outputs
        )
        self.trainer._call_custom_hooks("test_epoch_end", output_or_outputs)
