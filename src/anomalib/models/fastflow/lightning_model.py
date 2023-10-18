"""FastFlow Lightning Model Implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig, ListConfig
from torch import Tensor, optim

from anomalib.models.components import AnomalyModule
from anomalib.models.fastflow.loss import FastflowLoss
from anomalib.models.fastflow.torch_model import FastflowModel


class Fastflow(AnomalyModule):
    """PL Lightning Module for the FastFlow algorithm.

    Args:
        input_size (tuple[int, int]): Model input size.
        backbone (str): Backbone CNN network
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
        flow_steps (int, optional): Flow steps.
        conv3x3_only (bool, optinoal): Use only conv3x3 in fast_flow model. Defaults to False.
        hidden_ratio (float, optional): Ratio to calculate hidden var channels. Defaults to 1.0.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        backbone: str = "resnet18",
        pre_trained: bool = True,
        flow_steps: int = 8,
        conv3x3_only: bool = False,
        hidden_ratio: float = 1.0,
    ) -> None:
        super().__init__()

        self.model = FastflowModel(
            input_size=input_size,
            backbone=backbone,
            pre_trained=pre_trained,
            flow_steps=flow_steps,
            conv3x3_only=conv3x3_only,
            hidden_ratio=hidden_ratio,
        )
        self.loss = FastflowLoss()

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Perform the training step input and return the loss.

        Args:
            batch (batch: dict[str, str | Tensor]): Input batch
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

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Perform the validation step and return the anomaly map.

        Args:
            batch (dict[str, str | Tensor]): Input batch
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


class FastflowLightning(Fastflow):
    """PL Lightning Module for the FastFlow algorithm.

    Args:
        hparams (DictConfig | ListConfig): Model params
    """

    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        super().__init__(
            input_size=hparams.model.input_size,
            backbone=hparams.model.backbone,
            pre_trained=hparams.model.pre_trained,
            flow_steps=hparams.model.flow_steps,
            conv3x3_only=hparams.model.conv3x3_only,
            hidden_ratio=hparams.model.hidden_ratio,
        )
        self.hparams: DictConfig | ListConfig
        self.save_hyperparameters(hparams)
