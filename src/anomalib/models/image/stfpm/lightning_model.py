"""STFPM: Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection.

https://arxiv.org/abs/2103.04257
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.models.components import AnomalyModule
from anomalib.pre_processing import PreProcessor

from .loss import STFPMLoss
from .torch_model import STFPMModel

__all__ = ["Stfpm"]


class Stfpm(AnomalyModule):
    """PL Lightning Module for the STFPM algorithm.

    Args:
        backbone (str): Backbone CNN network
            Defaults to ``resnet18``.
        layers (list[str]): Layers to extract features from the backbone CNN
            Defaults to ``["layer1", "layer2", "layer3"]``.
        pre_processor (PreProcessor, optional): Pre-processor for the model.
            This is used to pre-process the input data before it is passed to the model.
            Defaults to ``None``.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: Sequence[str] = ("layer1", "layer2", "layer3"),
        pre_processor: PreProcessor | bool = True,
    ) -> None:
        super().__init__(pre_processor=pre_processor)

        self.model = STFPMModel(
            backbone=backbone,
            layers=layers,
        )
        self.loss = STFPMLoss()

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a training step of STFPM.

        For each batch, teacher and student and teacher features are extracted from the CNN.

        Args:
          batch (Batch): Input batch.
          args: Additional arguments.
          kwargs: Additional keyword arguments.

        Returns:
          Loss value
        """
        del args, kwargs  # These variables are not used.

        teacher_features, student_features = self.model.forward(batch.image)
        loss = self.loss(teacher_features, student_features)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a validation Step of STFPM.

        Similar to the training step, student/teacher features are extracted from the CNN for each batch, and
        anomaly map is computed.

        Args:
          batch (Batch): Input batch
          args: Additional arguments
          kwargs: Additional keyword arguments

        Returns:
          Dictionary containing images, anomaly maps, true labels and masks.
          These are required in `validation_epoch_end` for feature concatenation.
        """
        del args, kwargs  # These variables are not used.

        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Required trainer arguments."""
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers.

        Returns:
            Optimizer: SGD optimizer
        """
        return optim.SGD(
            params=self.model.student_model.parameters(),
            lr=0.4,
            momentum=0.9,
            dampening=0.0,
            weight_decay=0.001,
        )

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS
