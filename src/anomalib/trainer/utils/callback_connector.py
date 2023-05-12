"""Subclasses Lightning's callback connector to add necessary callbacks."""

import os
from datetime import timedelta
from typing import Dict, List, Optional

from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.trainer.connectors import callback_connector

from anomalib import trainer
from anomalib.utils.callbacks import TimerCallback


class CallbackConnector(callback_connector.CallbackConnector):
    """Subclasses Lightning's callback connector to add necessary callbacks."""

    def __init__(self, trainer: "trainer.AnomalibTrainer") -> None:
        self.trainer = trainer

    def on_trainer_init(
        self,
        callbacks: List[Callback] | Callback | None,
        enable_checkpointing: bool,
        enable_progress_bar: bool,
        default_root_dir: str | None,
        enable_model_summary: bool,
        max_time: str | timedelta | Dict[str, int] | None = None,
        accumulate_grad_batches: int | Dict[int, int] | None = None,
    ) -> None:
        """Adds the timer callback to the list of callbacks."""
        if isinstance(callbacks, list):
            callbacks.append(TimerCallback())
        elif isinstance(callbacks, Callback):
            callbacks = [callbacks, TimerCallback()]
        else:
            callbacks = TimerCallback()

        return super().on_trainer_init(
            callbacks,
            enable_checkpointing,
            enable_progress_bar,
            default_root_dir,
            enable_model_summary,
            max_time,
            accumulate_grad_batches,
        )

    def _configure_checkpoint_callbacks(self, enable_checkpointing: bool) -> None:
        if enable_checkpointing:
            early_stopping_callback: Optional[Callback] = None
            # remove model checkpoint if present
            # This is because the trainer class calls super().init() which in turn creates a CallbackConnector object
            # which adds a ModelCheckpoint callback. We need to remove this callback and add our own.
            self.trainer.callbacks = [
                callback for callback in self.trainer.callbacks if not isinstance(callback, ModelCheckpoint)
            ]
            # get early stopping callback if present
            for callback in self.trainer.callbacks:
                if isinstance(callback, EarlyStopping):
                    early_stopping_callback = callback
                    break
            monitor_metric = None if early_stopping_callback is None else early_stopping_callback.monitor
            monitor_mode = "max" if early_stopping_callback is None else early_stopping_callback.mode
            self.trainer.callbacks.append(
                ModelCheckpoint(
                    dirpath=os.path.join(self.trainer.default_root_dir, "weights", "lightning"),
                    filename="model",
                    monitor=monitor_metric,
                    mode=monitor_mode,
                    auto_insert_metric_name=False,
                    save_on_train_epoch_end=False,  # need to set this as validation loop now runs after train loop.
                )
            )
