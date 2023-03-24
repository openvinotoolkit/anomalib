"""Predict loop."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from lightning_fabric.utilities import move_data_to_device
from pytorch_lightning.loops.dataloader.prediction_loop import PredictionLoop
from pytorch_lightning.loops.epoch.prediction_epoch_loop import PredictionEpochLoop

import anomalib.trainer as core


class AnomalibPredictionEpochLoop(PredictionEpochLoop):
    """Predict epoch loop."""

    def __init__(self) -> None:
        super().__init__()
        self.trainer: core.AnomalibTrainer

    def _predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """Overrides the predict step of the base class.

        Everything is same except for the custom methods called after trainer's predict step.

        Note:
            Update methods only between the commented block.

        Warning:
            The order of callbacks and lightning module's ``predict_step`` has been swapped. Originally callbacks were
            called before the lightning step. But, we need to compute masks and labels before visualizer callback is
            called. So, in this case the lightning module's ``predict_step`` is called first, then custom methods for
            post-processing, and finally, the callbacks are called.


        Runs the actual predict step together with all the necessary bookkeeping and the hooks tied to the
        predict step.

        Args:
            batch: the current batch to run the prediction on
            batch_idx: the index of the current batch
            dataloader_idx: the index of the dataloader producing the current batch
        """
        # configure step_kwargs
        step_kwargs = self._build_kwargs(batch, batch_idx, dataloader_idx)

        # extract batch_indices and store them
        batch_indices = self._get_batch_indices(dataloader_idx)
        self.current_batch_indices = batch_indices[batch_idx] if batch_indices else []

        self.trainer._call_callback_hooks("on_predict_batch_start", batch, batch_idx, dataloader_idx)
        self.trainer._call_lightning_module_hook("on_predict_batch_start", batch, batch_idx, dataloader_idx)

        self.batch_progress.increment_started()

        predictions = self.trainer._call_strategy_hook("predict_step", *step_kwargs.values())

        self.batch_progress.increment_processed()

        if predictions is None:
            self._warning_cache.warn("predict returned None if it was on purpose, ignore this warning...")

        self.trainer._call_lightning_module_hook("on_predict_batch_end", predictions, batch, batch_idx, dataloader_idx)

        # Call custom methods on the predictions
        self.trainer.post_processor.compute_labels(predictions)
        self.trainer.post_processor.apply_thresholding(self.trainer.lightning_module, predictions)
        self.trainer.normalizer.normalize(self.trainer.lightning_module, predictions)
        # --------------------------------------

        self.trainer._call_callback_hooks("on_predict_batch_end", predictions, batch, batch_idx, dataloader_idx)

        self.batch_progress.increment_completed()

        if self.should_store_predictions:
            self.predictions.append(move_data_to_device(predictions, torch.device("cpu")))


class AnomalibPredictionLoop(PredictionLoop):
    """Prediction loop."""

    def __init__(self) -> None:
        super().__init__()
        self.trainer: core.AnomalibTrainer

    def on_run_start(self) -> None:
        """Setup epoch loop."""
        epoch_loop = AnomalibPredictionEpochLoop()
        epoch_loop.trainer = self.trainer
        epoch_loop.return_predictions = self._return_predictions
        self.connect(epoch_loop=epoch_loop)
        return super().on_run_start()
