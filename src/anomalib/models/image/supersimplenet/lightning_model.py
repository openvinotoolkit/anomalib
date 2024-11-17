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

from .loss import SSNLoss
from .torch_model import SuperSimpleNetModel


class SuperSimpleNet(AnomalyModule):
    """PL Lightning Module for the SuperSimpleNet model.

    Args:
        perlin_threshold (float): threshold value for Perlin noise thresholding during anomaly generation.
        backbone (str): backbone name
        layers (list[str]): backbone layers utilised
        supervised (bool): whether the model will be trained in supervised mode. False by default (unsupervised).
    """

    def __init__(
        self,
        perlin_threshold: float = 0.2,
        backbone: str = "wide_resnet50_2",
        layers: list[str] = ["layer2", "layer3"],  # noqa: B006
        supervised: bool = False,
    ) -> None:
        super().__init__()
        self.supervised = supervised
        # stop grad in unsupervised
        if supervised:
            stop_grad = False
            self.norm_clip_val = 1
        else:
            stop_grad = True
            self.norm_clip_val = 0

        self.model = SuperSimpleNetModel(
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

        anomaly_map, anomaly_score, masks, labels = self.model(
            images=batch.image, masks=batch.gt_mask, labels=batch.gt_label
        )
        loss = self.loss(pred_map=anomaly_map, pred_score=anomaly_score, target_mask=masks, target_label=labels)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform the validation step and return the anomaly map and anomaly score.

        Args:
            batch (dict[str, str | torch.Tensor]): Input batch
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            STEP_OUTPUT | None: batch dictionary containing anomaly-maps.
        """
        # These variables are not used.
        del args, kwargs

        # Get anomaly maps and predicted scores from the model.
        predictions = self.model(batch.image)

        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return SuperSimpleNet trainer arguments."""
        return {"gradient_clip_val": self.norm_clip_val, "num_sanity_val_steps": 0}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        pass

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS
