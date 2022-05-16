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
from typing import List, Tuple

import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch import optim

from anomalib.models.components import AnomalyModule
from anomalib.models.stfpm.torch_model import STFPMModel

logger = logging.getLogger(__name__)

__all__ = ["StfpmLightning"]


class StfpmLightning(AnomalyModule):
    """PL Lightning Module for the STFPM algorithm.

    Args:
        adaptive_threshold (bool): Boolean to automatically choose adaptive threshold
        default_image_threshold (float): Manual default image threshold
        default_pixel_threshold (float): Manaul default pixel threshold
        input_size (Tuple[int, int]): Size of the model input.
        backbone (str): Backbone CNN network
        layers (List[str]): Layers to extract features from the backbone CNN
        learning_rate (float, optional): Learning rate. Defaults to 0.4.
        momentum (float, optional): Momentum. Defaults to 0.9.
        weight_decay (float, optional): Weight decay. Defaults to 0.0001.
        early_stopping_metric (str, optional): Early stopping metric. Defaults to "pixel_AUROC".
        early_stopping_patience (int, optional): Early stopping patience. Defaults to 3.
        early_stopping_mode (str, optional): Early stopping mode. Defaults to "max".
    """

    def __init__(
        self,
        adaptive_threshold: bool,
        default_image_threshold: float,
        default_pixel_threshold: float,
        input_size: Tuple[int, int],
        backbone: str,
        layers: List[str],
        learning_rate: float = 0.4,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        early_stopping_metric: str = "pixel_AUROC",
        early_stopping_patience: int = 3,
        early_stopping_mode: str = "max",
    ):

        super().__init__(
            adaptive_threshold=adaptive_threshold,
            default_image_threshold=default_image_threshold,
            default_pixel_threshold=default_pixel_threshold,
        )
        logger.info("Initializing Stfpm Lightning model.")

        self.model = STFPMModel(
            input_size=input_size,
            backbone=backbone,
            layers=layers,
        )
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_mode = early_stopping_mode
        self.loss_val = 0

    def configure_callbacks(self):
        """Configure model-specific callbacks."""
        early_stopping = EarlyStopping(
            monitor=self.early_stopping_metric,
            patience=self.early_stopping_patience,
            mode=self.early_stopping_mode,
        )
        return [early_stopping]

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers by creating an SGD optimizer.

        Returns:
            (Optimizer): SGD optimizer
        """
        return optim.SGD(
            params=self.model.student_model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
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
