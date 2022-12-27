"""FastFlow Lightning Model Implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

from pytorch_lightning.utilities.cli import MODEL_REGISTRY

from anomalib.models.components import AnomalyModule
from anomalib.models.fastflow.loss import FastflowLoss
from anomalib.models.fastflow.torch_model import FastflowModel


@MODEL_REGISTRY
class Fastflow(AnomalyModule):
    """PL Lightning Module for the FastFlow algorithm.

    Args:
        input_size (Tuple[int, int]): Model input size.
        backbone (str): Backbone CNN network
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
        flow_steps (int, optional): Flow steps.
        conv3x3_only (bool, optinoal): Use only conv3x3 in fast_flow model. Defaults to False.
        hidden_ratio (float, optional): Ratio to calculate hidden var channels. Defaults to 1.0.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        backbone: str,
        pre_trained: bool = True,
        flow_steps: int = 8,
        conv3x3_only: bool = False,
        hidden_ratio: float = 1.0,
    ):
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

    def training_step(self, batch, _):  # pylint: disable=arguments-differ
        """Forward-pass input and return the loss.

        Args:
            batch (Tensor): Input batch
            _batch_idx: Index of the batch.

        Returns:
            STEP_OUTPUT: Dictionary containing the loss value.
        """
        hidden_variables, jacobians = self.model(batch["image"])
        loss = self.loss(hidden_variables, jacobians)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, _):  # pylint: disable=arguments-differ
        """Forward-pass the input and return the anomaly map.

        Args:
            batch (Tensor): Input batch
            _batch_idx: Index of the batch.

        Returns:
            dict: batch dictionary containing anomaly-maps.
        """
        anomaly_maps = self.model(batch["image"])
        batch["anomaly_maps"] = anomaly_maps
        return batch
