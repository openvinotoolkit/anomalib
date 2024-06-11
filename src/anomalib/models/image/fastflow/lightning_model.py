"""FastFlow Lightning Model Implementation.

https://arxiv.org/abs/2111.07677
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim

from anomalib import LearningType
from anomalib.models.components import AnomalyModule

from .loss import FastflowLoss
from .torch_model import FastflowModel


class Fastflow(AnomalyModule):
    """PL Lightning Module for the FastFlow algorithm.

    Args:
        backbone (str): Backbone CNN network
            Defaults to ``resnet18``.
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
            Defaults to ``True``.
        flow_steps (int, optional): Flow steps.
            Defaults to ``8``.
        conv3x3_only (bool, optinoal): Use only conv3x3 in fast_flow model.
            Defaults to ``False``.
        hidden_ratio (float, optional): Ratio to calculate hidden var channels.
            Defaults to ``1.0`.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pre_trained: bool = True,
        flow_steps: int = 8,
        conv3x3_only: bool = False,
        hidden_ratio: float = 1.0,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.pre_trained = pre_trained
        self.flow_steps = flow_steps
        self.conv3x3_only = conv3x3_only
        self.hidden_ratio = hidden_ratio

        self.model: FastflowModel
        self.loss = FastflowLoss()

    def _setup(self) -> None:
        if self.input_size is None:
            msg = "Fastflow needs input size to build torch model."
            raise ValueError(msg)

        self.model = FastflowModel(
            input_size=self.input_size,
            backbone=self.backbone,
            pre_trained=self.pre_trained,
            flow_steps=self.flow_steps,
            conv3x3_only=self.conv3x3_only,
            hidden_ratio=self.hidden_ratio,
        )

    def training_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Perform the training step input and return the loss.

        Args:
            batch (batch: dict[str, str | torch.Tensor]): Input batch
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            STEP_OUTPUT: Dictionary containing the loss value.
        """
        del args, kwargs  # These variables are not used.

        hidden_variables, jacobians = self.model(batch["image"])
        loss = self.loss(hidden_variables, jacobians)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Perform the validation step and return the anomaly map.

        Args:
            batch (dict[str, str | torch.Tensor]): Input batch
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            STEP_OUTPUT | None: batch dictionary containing anomaly-maps.
        """
        del args, kwargs  # These variables are not used.

        anomaly_maps = self.model(batch["image"])
        batch["anomaly_maps"] = anomaly_maps
        return batch

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return FastFlow trainer arguments."""
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers for each decoder.

        Returns:
            Optimizer: Adam optimizer for each decoder
        """
        return optim.Adam(
            params=self.model.parameters(),
            lr=0.001,
            weight_decay=0.00001,
        )

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS
