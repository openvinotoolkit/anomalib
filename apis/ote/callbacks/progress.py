"""
Progressbar Callback for OTE task
"""

# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import Optional, Union

from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.train_parameters import TrainParameters, default_progress_callback
from pytorch_lightning.callbacks.progress import ProgressBar


class ProgressCallback(ProgressBar):
    """
    Modifies progress callback to show completion of the entire training step
    """

    def __init__(self, parameters: Optional[Union[TrainParameters, InferenceParameters]] = None) -> None:
        super().__init__()
        self.current_epoch: int = 0
        self.max_epochs: int = 0
        self._progress: float = 0

        if parameters is not None:
            self.update_progress_callback = parameters.update_progress
        else:
            self.update_progress_callback = default_progress_callback

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.current_epoch = trainer.current_epoch
        self.max_epochs = trainer.max_epochs
        self._reset_progress()

    def on_validation_start(self, trainer, pl_module):
        super().on_validation_start(trainer, pl_module)
        self._reset_progress()

    def on_predict_start(self, trainer, pl_module):
        super().on_predict_start(trainer, pl_module)
        self._reset_progress()

    def on_test_start(self, trainer, pl_module):
        super().on_test_start(trainer, pl_module)
        self._reset_progress()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """
        Adds training completion percentage to the progress bar
        """
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.current_epoch = trainer.current_epoch
        self._update_progress(stage="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self._update_progress(stage="val")

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_predict_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self._update_progress(stage="predict")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """
        Adds testing completion percentage to the progress bar
        """
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self._update_progress(stage="test")

    def _reset_progress(self):
        self._progress = 0.0

    def _get_progress(self, stage: str = "train"):
        """
        Get progress for train and test stages.

        Args:
            stage (str, optional): Train or Test stages. Defaults to "train".
        """

        if stage == "train":
            # Progress is calculated on the upper bound (max epoch).
            # Early stopping might stop the training before the progress reaches 100%
            self._progress = (
                (self.train_batch_idx + self.current_epoch * self.total_train_batches)
                / (self.total_train_batches * self.max_epochs)
            ) * 100

        elif stage == "val":
            self._progress = (self.val_batch_idx / self.total_val_batches) * 100

        elif stage == "predict":
            self._progress = (self.predict_batch_idx / self.total_predict_batches) * 100

        elif stage == "test":
            self._progress = (self.test_batch_idx / self.total_test_batches) * 100
        else:
            raise ValueError(f"Unknown stage {stage}. Available: train, val, predict and test")

        return self._progress

    def _update_progress(self, stage: str):
        progress = self._get_progress(stage)
        self.update_progress_callback(progress)
