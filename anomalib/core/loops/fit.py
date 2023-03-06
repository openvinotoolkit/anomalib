from __future__ import annotations

from pytorch_lightning.loops.fit_loop import FitLoop

import anomalib.core as core

from .validate import AnomalibValidationLoop


class AnomalibFitLoop(FitLoop):
    def __init__(self, min_epochs: int | None, max_epochs: int | None) -> None:
        super().__init__(min_epochs=min_epochs, max_epochs=max_epochs)
        self.trainer: "core.AnomalibTrainer"

    def on_run_start(self) -> None:
        self.epoch_loop.replace(val_loop=AnomalibValidationLoop())
        self.trainer._call_custom_hooks("on_run_start")
        return super().on_run_start()
