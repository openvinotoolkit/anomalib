from typing import Any

import torch
from lightning_fabric.utilities import move_data_to_device
from pytorch_lightning.loops.dataloader.prediction_loop import PredictionLoop
from pytorch_lightning.loops.epoch.prediction_epoch_loop import PredictionEpochLoop


class AnomalibPredictionEpochLoop(PredictionEpochLoop):
    def on_advance_end(self) -> None:
        """Runs at the end of ``predict_step``."""
        self.trainer._call_custom_hooks("predict_step", self.predictions)

    # def _predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
    #     """Override the predict step as we need to post_process the outputs.

    #     This method is same as the base class except for one additional line that calls custom callbacks.
    #     """
    #     # configure step_kwargs
    #     step_kwargs = self._build_kwargs(batch, batch_idx, dataloader_idx)

    #     # extract batch_indices and store them
    #     batch_indices = self._get_batch_indices(dataloader_idx)
    #     self.current_batch_indices = batch_indices[batch_idx] if batch_indices else []

    #     self.trainer._call_callback_hooks("on_predict_batch_start", batch, batch_idx, dataloader_idx)
    #     self.trainer._call_lightning_module_hook("on_predict_batch_start", batch, batch_idx, dataloader_idx)

    #     self.batch_progress.increment_started()

    #     predictions = self.trainer._call_strategy_hook("predict_step", *step_kwargs.values())

    #     self.batch_progress.increment_processed()

    #     if predictions is None:
    #         self._warning_cache.warn("predict returned None if it was on purpose, ignore this warning...")

    #     self.trainer._call_callback_hooks("on_predict_batch_end", predictions, batch, batch_idx, dataloader_idx)
    #     self.trainer._call_lightning_module_hook("on_predict_batch_end", predictions, batch, batch_idx, dataloader_idx)

    #     # Call custom loops.
    #     self.trainer._call_custom_hooks("predict_step", predictions)

    #     self.batch_progress.increment_completed()

    #     if self.should_store_predictions:
    #         self.predictions.append(move_data_to_device(predictions, torch.device("cpu")))


class AnomalibPredictionLoop(PredictionLoop):
    def on_run_start(self) -> None:
        epoch_loop = AnomalibPredictionEpochLoop()
        epoch_loop.trainer = self.trainer
        epoch_loop.return_predictions = self._return_predictions
        self.connect(epoch_loop=epoch_loop)
        return super().on_run_start()
