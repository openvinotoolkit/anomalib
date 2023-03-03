from typing import Any

import torch
from lightning_fabric.utilities import move_data_to_device
from pytorch_lightning.loops.dataloader.prediction_loop import PredictionLoop
from pytorch_lightning.loops.epoch.prediction_epoch_loop import PredictionEpochLoop


class AnomalibPredictionEpochLoop(PredictionEpochLoop):
    def on_advance_end(self) -> None:
        """Runs at the end of ``predict_step``."""
        self.trainer._call_custom_hooks("predict_step", self.predictions)


class AnomalibPredictionLoop(PredictionLoop):
    def on_run_start(self) -> None:
        epoch_loop = AnomalibPredictionEpochLoop()
        epoch_loop.trainer = self.trainer
        epoch_loop.return_predictions = self._return_predictions
        self.connect(epoch_loop=epoch_loop)
        return super().on_run_start()
