"""Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection.

https://arxiv.org/pdf/2110.02855.pdf
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, Tuple

from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from torch import Tensor

from anomalib.models.components import AnomalyModule

from .loss import CsFlowLoss
from .torch_model import CsFlowModel

logger = logging.getLogger(__name__)

__all__ = ["Csflow"]


@MODEL_REGISTRY
class Csflow(AnomalyModule):
    """Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection.

    Args:
        input_size (Tuple[int, int]): Size of the model input.
        n_coupling_blocks (int): Number of coupling blocks in the model.
        cross_conv_hidden_channels (int): Number of hidden channels in the cross convolution.
        clamp (int): Clamp value for glow layer.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        cross_conv_hidden_channels: int,
        n_coupling_blocks: int,
        clamp: int,
    ):
        super().__init__()
        self.model: CsFlowModel = CsFlowModel(
            input_size=input_size,
            cross_conv_hidden_channels=cross_conv_hidden_channels,
            n_coupling_blocks=n_coupling_blocks,
            clamp=clamp,
        )
        self.loss = CsFlowLoss()

    def training_step(self, batch, _) -> Dict[str, Tensor]:
        """Training Step of CS-Flow.

        Args:
            batch (Tensor): Input batch
            _: Index of the batch.

        Returns:
            Loss value
        """
        self.model.feature_extractor.eval()
        z_dist, jacobians = self.model(batch["image"])
        loss = self.loss(z_dist, jacobians)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, _) -> Dict[str, Tensor]:
        """Validation step for CS Flow.

        Args:
            batch (Tensor): Input batch

        Returns:
            Dict[str, Tensor]: Dictionary containing the anomaly map, scores, etc.
        """
        anomaly_maps, anomaly_scores = self.model(batch["image"])
        batch["anomaly_maps"] = anomaly_maps
        batch["pred_scores"] = anomaly_scores
        return batch
