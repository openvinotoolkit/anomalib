"""
Model monitor callback for OTE training
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

import copy
import logging
from typing import Callable, Optional

import pytorch_lightning as pl
from ote_sdk.entities.metrics import NullPerformance, Performance, ScoreMetric
from ote_sdk.entities.model import ModelEntity
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


class ModelMonitorCallback(Callback):
    """
    Callback to check if model has improved compared to the previous version.
    """

    def __init__(self, output_model: ModelEntity, save_model_fn: Callable):
        self.output_model = output_model
        self.save_model_fn = save_model_fn

        self.old_model: Optional[pl.LightningModule] = None
        self.old_performance: Optional[float] = None

    def on_fit_start(self, _trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        When fit begins, save a copy of the old model that we can restore if training does not improve.
        """
        self.old_model = copy.deepcopy(pl_module)

    def on_sanity_check_end(self, _trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        When the validation sanity check ends, store the performance of the previous model on the new validation set.
        """
        self.old_performance = pl_module.results.performance["image_f1_score"]

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        When fit ends, compare the validation performance to the performance of the previous model. If it has improved,
        save the new model. Otherwise discard the new model and restore the copy of the old one.
        """
        new_performance = pl_module.results.performance["image_f1_score"]
        is_improved = self.old_performance < new_performance
        is_first_training = self.output_model.performance == NullPerformance()
        if is_improved or is_first_training:
            if is_first_training:
                logger.info("First training round, saving the model.")
            else:
                logger.info("Training finished, and it has an improved model")
            performance = Performance(score=ScoreMetric(name="F1 Score", value=new_performance))
            self.save_model_fn(self.output_model)
            self.output_model.performance = performance
        else:
            logger.info("Model performance has not improved while training. No new model has been saved.")
            trainer.model = self.old_model
