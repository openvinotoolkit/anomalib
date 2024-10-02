"""Lightning Implementatation of the CFA Model.

CFA: Coupled-hypersphere-based Feature Adaptation for Target-Oriented Anomaly Localization

Paper https://arxiv.org/abs/2206.04325
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib import LearningType
from anomalib.models.components import AnomalyModule

from .loss import CfaLoss
from .torch_model import CfaModel

logger = logging.getLogger(__name__)

__all__ = ["Cfa"]


class Cfa(AnomalyModule):
    """CFA: Coupled-hypersphere-based Feature Adaptation for Target-Oriented Anomaly Localization.

    Args:
        backbone (str): Backbone CNN network
            Defaults to ``"wide_resnet50_2"``.
        gamma_c (int, optional): gamma_c value from the paper.
            Defaults to ``1``.
        gamma_d (int, optional): gamma_d value from the paper.
            Defaults to ``1``.
        num_nearest_neighbors (int): Number of nearest neighbors.
            Defaults to ``3``.
        num_hard_negative_features (int): Number of hard negative features.
            Defaults to ``3``.
        radius (float): Radius of the hypersphere to search the soft boundary.
            Defaults to ``1e-5``.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        gamma_c: int = 1,
        gamma_d: int = 1,
        num_nearest_neighbors: int = 3,
        num_hard_negative_features: int = 3,
        radius: float = 1e-5,
    ) -> None:
        super().__init__()
        self.model: CfaModel = CfaModel(
            backbone=backbone,
            gamma_c=gamma_c,
            gamma_d=gamma_d,
            num_nearest_neighbors=num_nearest_neighbors,
            num_hard_negative_features=num_hard_negative_features,
            radius=radius,
        )
        self.loss = CfaLoss(
            num_nearest_neighbors=num_nearest_neighbors,
            num_hard_negative_features=num_hard_negative_features,
            radius=radius,
        )

    def on_train_start(self) -> None:
        """Initialize the centroid for the memory bank computation."""
        self.model.initialize_centroid(data_loader=self.trainer.datamodule.train_dataloader())

    def training_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Perform the training step for the CFA model.

        Args:
            batch (dict[str, str | torch.Tensor]): Batch input.
            *args: Arguments.
            **kwargs: Keyword arguments.

        Returns:
            STEP_OUTPUT: Loss value.
        """
        del args, kwargs  # These variables are not used.

        distance = self.model(batch["image"])
        loss = self.loss(distance)
        return {"loss": loss}

    def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Perform the validation step for the CFA model.

        Args:
            batch (dict[str, str | torch.Tensor]): Input batch.
            *args: Arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict: Anomaly map computed by the model.
        """
        del args, kwargs  # These variables are not used.

        batch["anomaly_maps"] = self.model(batch["image"])
        return batch

    @staticmethod
    def backward(loss: torch.Tensor, *args, **kwargs) -> None:
        """Perform backward-pass for the CFA model.

        Args:
            loss (torch.Tensor): Loss value.
            *args: Arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs  # These variables are not used.

        # TODO(samet-akcay): Investigate why retain_graph is needed.
        # CVS-122673
        loss.backward(retain_graph=True)

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """CFA specific trainer arguments."""
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers for the CFA Model.

        Returns:
            Optimizer: Adam optimizer for each decoder
        """
        return torch.optim.AdamW(
            params=self.model.parameters(),
            lr=1e-3,
            weight_decay=5e-4,
            amsgrad=True,
        )

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS
