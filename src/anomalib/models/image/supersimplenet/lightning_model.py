"""SuperSimpleNet: Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection.

Paper https://arxiv.org/pdf/2408.03143
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.models import AnomalyModule
from anomalib.models.image.supersimplenet.loss import SSNLoss


class SuperSimpleNet(AnomalyModule):
    """PL Lightning Module for the SuperSimpleNet model.

    Args:
        perlin_threshold (float): threshold value for Perlin noise thresholding during anomaly generation.
        backbone (str): backbone name
        layers (list[str]): backbone layers utilised
        stop_grad (bool): whether to stop gradient from class. to seg. head.
    """

    def __init__(
        self,
        perlin_threshold: float = 0.2,
        backbone: str = "wide_resnet50_2",
        layers: list[str] = ["layer2", "layer3"],  # noqa: B006
        stop_grad: bool = True,
    ) -> None:
        super().__init__()
        self.model = SuperSimpleNet(
            perlin_threshold=perlin_threshold,
            backbone=backbone,
            layers=layers,
            stop_grad=stop_grad,
        )
        self.loss = SSNLoss()

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform the training step input and return the loss.

        Args:
            batch (batch: dict[str, str | torch.Tensor]): Input batch
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            STEP_OUTPUT: Dictionary containing the loss value.
        """
        del args, kwargs  # These variables are not used.

        anomaly_map, anomaly_score, masks, labels = self.model(batch.image)
        loss = self.loss(pred_map=anomaly_map, pred_score=anomaly_score, target_mask=masks, target_label=labels)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        pass

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        # TODO - normclip
        pass

    def configure_optimizers(self) -> OptimizerLRScheduler:
        pass

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS
