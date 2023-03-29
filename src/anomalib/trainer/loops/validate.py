"""Validation loop."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import lru_cache
from typing import List

from pytorch_lightning.loops.dataloader.evaluation_loop import EvaluationLoop
from pytorch_lightning.loops.epoch.evaluation_epoch_loop import EvaluationEpochLoop
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT

import anomalib.trainer as core


class AnomalibValidationEpochLoop(EvaluationEpochLoop):
    def __init__(self):
        super().__init__()
        self.trainer: core.AnomalibTrainer

    def _evaluation_step_end(self, *args, **kwargs) -> STEP_OUTPUT | None:
        """Post-processes outputs and updates normalization metrics after the end of one validation step end."""
        outputs = super()._evaluation_step_end(*args, **kwargs)
        if outputs is not None:
            self.trainer.post_processor.apply_predictions(outputs)
            self.trainer.thresholder.update(outputs)
            self.trainer.normalizer.update_metrics(outputs)
        return outputs

    @lru_cache(1)
    def _should_track_batch_outputs_for_epoch_end(self) -> bool:
        """Track batch outputs for epoch end.

        If this is not overridden, the outputs are not collected if the model does not have a ``validation_epoch_end``
        method.Lightning ends up deleting epoch outputs if this is false. This ensures that we have the outputs when
        computing the metrics on the validation set.
        """
        return True


class AnomalibValidationLoop(EvaluationLoop):
    def __init__(self) -> None:
        super().__init__()
        self.trainer: core.AnomalibTrainer
        self.epoch_loop = AnomalibValidationEpochLoop()

    def _evaluation_epoch_end(self, outputs: List[EPOCH_OUTPUT]):
        """Runs ``validation_epoch_end``

        Args:
            outputs (List[EPOCH_OUTPUT]): Outputs
        """
        super()._evaluation_epoch_end(outputs)
        output_or_outputs: EPOCH_OUTPUT | List[EPOCH_OUTPUT] = (
            outputs[0] if len(outputs) > 0 and self.num_dataloaders == 1 else outputs
        )

        # keep custom code between these lines
        self.trainer.thresholder.compute()
        self.trainer.metrics_manager.set_threshold()
        self.trainer.metrics_manager.compute(output_or_outputs)
        self.trainer.metrics_manager.log(self.trainer, "validation_epoch_end")
