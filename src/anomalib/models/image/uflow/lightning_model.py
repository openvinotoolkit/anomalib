"""U-Flow: A U-shaped Normalizing Flow for Anomaly Detection with Unsupervised Threshold.

https://arxiv.org/pdf/2211.12353.pdf
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler

from anomalib import LearningType
from anomalib.models.components import AnomalyModule

from .loss import UFlowLoss
from .torch_model import UflowModel

__all__ = ["Uflow"]


class Uflow(AnomalyModule):
    """PL Lightning Module for the UFLOW algorithm."""

    def __init__(
        self,
        input_size: tuple[int, int] = (448, 448),
        backbone: str = "mcait",
        flow_steps: int = 4,
        affine_clamp: float = 2.0,
        affine_subnet_channels_ratio: float = 1.0,
        permute_soft: bool = False,
    ) -> None:
        """Uflow model.

        Args:
            input_size (tuple[int, int]): Input image size.
            backbone (str): Backbone name.
            flow_steps (int): Number of flow steps.
            affine_clamp (float): Affine clamp.
            affine_subnet_channels_ratio (float): Affine subnet channels ratio.
            permute_soft (bool): Whether to use soft permutation.
        """
        super().__init__()
        self.model: UflowModel = UflowModel(
            input_size=input_size,
            backbone=backbone,
            flow_steps=flow_steps,
            affine_clamp=affine_clamp,
            affine_subnet_channels_ratio=affine_subnet_channels_ratio,
            permute_soft=permute_soft,
        )
        self.loss = UFlowLoss()

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:  # noqa: ARG002 | unused arguments
        """Training step."""
        z, ljd = self.model(batch["image"])
        loss = self.loss(z, ljd)
        self.log_dict({"loss": loss}, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:  # noqa: ARG002 | unused arguments
        """Validation step."""
        anomaly_maps = self.model(batch["image"])
        batch["anomaly_maps"] = anomaly_maps
        return batch

    def configure_optimizers(self) -> tuple[list[LightningOptimizer], list[LRScheduler]]:
        """Return optimizer and scheduler."""
        # Optimizer
        # values used in paper: bottle: 0.0001128999, cable: 0.0016160391, capsule: 0.0012118892, carpet: 0.0012118892,
        # grid: 0.0000362248, hazelnut: 0.0013268899, leather: 0.0006124724, metal_nut: 0.0008148858,
        # pill: 0.0010756100, screw: 0.0004155987, tile: 0.0060457548, toothbrush: 0.0001287313,
        # transistor: 0.0011212904, wood: 0.0002466546, zipper: 0.0000455247
        optimizer = torch.optim.Adam([{"params": self.parameters(), "initial_lr": 1e-3}], lr=1e-3, weight_decay=1e-5)

        # Scheduler for slowly reducing learning rate
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.4,
            total_iters=25000,
        )
        return [optimizer], [scheduler]

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return EfficientAD trainer arguments."""
        return {"num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS
