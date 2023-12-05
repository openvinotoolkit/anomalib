"""Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection.

https://arxiv.org/pdf/2110.02855.pdf
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib.models.components import AnomalyModule

from .loss import CsFlowLoss
from .torch_model import CsFlowModel

logger = logging.getLogger(__name__)

__all__ = ["Csflow"]


class Csflow(AnomalyModule):
    """Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection.

    Args:
        input_size (tuple[int, int]): Size of the model input.
            Defaults to ``(256, 256)``.
        n_coupling_blocks (int): Number of coupling blocks in the model.
            Defaults to ``4``.
        cross_conv_hidden_channels (int): Number of hidden channels in the cross convolution.
            Defaults to ``1024``.
        clamp (int): Clamp value for glow layer.
            Defaults to ``3``.
        num_channels (int): Number of channels in the model.
            Defaults to ``3``.
    """

    def __init__(
        self,
        input_size: tuple[int, int] = (256, 256),
        cross_conv_hidden_channels: int = 1024,
        n_coupling_blocks: int = 4,
        clamp: int = 3,
        num_channels: int = 3,
    ) -> None:
        super().__init__()
        self.model: CsFlowModel = CsFlowModel(
            input_size=input_size,
            cross_conv_hidden_channels=cross_conv_hidden_channels,
            n_coupling_blocks=n_coupling_blocks,
            clamp=clamp,
            num_channels=num_channels,
        )
        self.loss = CsFlowLoss()

    def training_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Perform the training step of CS-Flow.

        Args:
            batch (dict[str, str | torch.Tensor]): Input batch
            args: Arguments.
            kwargs: Keyword arguments.

        Returns:
            Loss value
        """
        del args, kwargs  # These variables are not used.

        self.model.feature_extractor.eval()
        z_dist, jacobians = self.model(batch["image"])
        loss = self.loss(z_dist, jacobians)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Perform the validation step for CS Flow.

        Args:
            batch (torch.Tensor): Input batch
            args: Arguments.
            kwargs: Keyword arguments.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing the anomaly map, scores, etc.
        """
        del args, kwargs  # These variables are not used.

        anomaly_maps, anomaly_scores = self.model(batch["image"])
        batch["anomaly_maps"] = anomaly_maps
        batch["pred_scores"] = anomaly_scores
        return batch

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """CS-Flow-specific trainer arguments."""
        return {"gradient_clip_val": 1, "num_sanity_val_steps": 0}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers.

        Returns:
            Optimizer: Adam optimizer
        """
        return torch.optim.Adam(
            self.parameters(),
            lr=2e-4,
            eps=1e-04,
            weight_decay=1e-5,
            betas=(0.5, 0.9),
        )
