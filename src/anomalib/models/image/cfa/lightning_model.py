"""Lightning Implementation of the CFA Model.

CFA: Coupled-hypersphere-based Feature Adaptation for Target-Oriented Anomaly
Localization.

Paper: https://arxiv.org/abs/2206.04325

This implementation uses PyTorch Lightning for training and inference.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .loss import CfaLoss
from .torch_model import CfaModel

logger = logging.getLogger(__name__)

__all__ = ["Cfa"]


class Cfa(AnomalibModule):
    """CFA Lightning Module.

    The CFA model performs anomaly detection and localization using coupled
    hypersphere-based feature adaptation.

    Args:
        backbone (str): Name of the backbone CNN network.
            Defaults to ``"wide_resnet50_2"``.
        gamma_c (int, optional): Centroid loss weight parameter.
            Defaults to ``1``.
        gamma_d (int, optional): Distance loss weight parameter.
            Defaults to ``1``.
        num_nearest_neighbors (int): Number of nearest neighbors to consider.
            Defaults to ``3``.
        num_hard_negative_features (int): Number of hard negative features to use.
            Defaults to ``3``.
        radius (float): Radius of the hypersphere for soft boundary search.
            Defaults to ``1e-5``.
        pre_processor (PreProcessor | bool, optional): Pre-processor instance or
            boolean flag.
            Defaults to ``True``.
        post_processor (PostProcessor | bool, optional): Post-processor instance or
            boolean flag.
            Defaults to ``True``.
        evaluator (Evaluator | bool, optional): Evaluator instance or boolean flag.
            Defaults to ``True``.
        visualizer (Visualizer | bool, optional): Visualizer instance or boolean
            flag.
            Defaults to ``True``.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        gamma_c: int = 1,
        gamma_d: int = 1,
        num_nearest_neighbors: int = 3,
        num_hard_negative_features: int = 3,
        radius: float = 1e-5,
        # Anomalib's Auxiliary Components
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
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
        """Initialize the centroid for memory bank computation.

        This method is called at the start of training to compute the initial
        centroid using the training data.
        """
        self.model.initialize_centroid(data_loader=self.trainer.datamodule.train_dataloader())

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a training step.

        Args:
            batch (Batch): Input batch containing images and metadata.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            STEP_OUTPUT: Dictionary containing the loss value.
        """
        del args, kwargs  # These variables are not used.

        distance = self.model(batch.image)
        loss = self.loss(distance)
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a validation step.

        Args:
            batch (Batch): Input batch containing images and metadata.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            STEP_OUTPUT: Batch object updated with model predictions.
        """
        del args, kwargs  # These variables are not used.

        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @staticmethod
    def backward(loss: torch.Tensor, *args, **kwargs) -> None:
        """Perform backward pass.

        Args:
            loss (torch.Tensor): Computed loss value.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Note:
            Uses ``retain_graph=True`` due to computational graph requirements.
            See CVS-122673 for more details.
        """
        del args, kwargs  # These variables are not used.

        # TODO(samet-akcay): Investigate why retain_graph is needed.
        # CVS-122673
        loss.backward(retain_graph=True)

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Get CFA-specific trainer arguments.

        Returns:
            dict[str, Any]: Dictionary containing trainer configuration:
                - ``gradient_clip_val``: Set to ``0`` to disable gradient clipping
                - ``num_sanity_val_steps``: Set to ``0`` to skip validation sanity
                  checks
        """
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer.

        Returns:
            torch.optim.Optimizer: AdamW optimizer configured with:
                - Learning rate: ``1e-3``
                - Weight decay: ``5e-4``
                - AMSGrad: ``True``
        """
        return torch.optim.AdamW(
            params=self.model.parameters(),
            lr=1e-3,
            weight_decay=5e-4,
            amsgrad=True,
        )

    @property
    def learning_type(self) -> LearningType:
        """Get the learning type.

        Returns:
            LearningType: Indicates this is a one-class classification model.
        """
        return LearningType.ONE_CLASS
