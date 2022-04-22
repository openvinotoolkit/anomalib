"""STFPM: Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection.

https://arxiv.org/abs/2103.04257
"""

# Copyright (C) 2020 Intel Corporation
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

import logging

import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch import optim

from anomalib.models.components import AnomalyModule
from anomalib.models.stfpm.torch_model import STFPMModel

logger = logging.getLogger(__name__)

__all__ = ["StfpmLightning"]


class StfpmLightning(AnomalyModule):
    """PL Lightning Module for the STFPM algorithm."""

    def __init__(self, hparams):
        super().__init__(hparams)
        logger.info("Initializing Stfpm Lightning model.")

        self.model = STFPMModel(
            layers=hparams.model.layers,
            input_size=hparams.model.input_size,
            tile_size=hparams.dataset.tiling.tile_size,
            tile_stride=hparams.dataset.tiling.stride,
            backbone=hparams.model.backbone,
            apply_tiling=hparams.dataset.tiling.apply,
        )
        self.loss_val = 0

    def configure_callbacks(self):
        """Configure model-specific callbacks."""
        early_stopping = EarlyStopping(
            monitor=self.hparams.model.early_stopping.metric,
            patience=self.hparams.model.early_stopping.patience,
            mode=self.hparams.model.early_stopping.mode,
        )
        return [early_stopping]

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers by creating an SGD optimizer.

        Returns:
            (Optimizer): SGD optimizer
        """
        return optim.SGD(
            params=self.model.student_model.parameters(),
            lr=self.hparams.model.lr,
            momentum=self.hparams.model.momentum,
            weight_decay=self.hparams.model.weight_decay,
        )

    def training_step(self, batch, _):  # pylint: disable=arguments-differ
        """Training Step of STFPM.

        For each batch, teacher and student and teacher features are extracted from the CNN.

        Args:
          batch (Tensor): Input batch
          _: Index of the batch.

        Returns:
          Hierarchical feature map
        """
        self.model.teacher_model.eval()
        teacher_features, student_features = self.model.forward(batch["image"])
        loss = self.loss_val + self.model.loss(teacher_features, student_features)
        self.loss_val = 0
        return {"loss": loss}

    def validation_step(self, batch, _):  # pylint: disable=arguments-differ
        """Validation Step of STFPM.

        Similar to the training step, student/teacher features are extracted from the CNN for each batch, and
        anomaly map is computed.

        Args:
          batch (Tensor): Input batch
          _: Index of the batch.

        Returns:
          Dictionary containing images, anomaly maps, true labels and masks.
          These are required in `validation_epoch_end` for feature concatenation.
        """
        batch["anomaly_maps"] = self.model(batch["image"])

        return batch
