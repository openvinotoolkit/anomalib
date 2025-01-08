"""Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection.

Paper: https://arxiv.org/pdf/2110.02855.pdf

This module provides the CS-Flow model implementation for anomaly detection.
CS-Flow uses normalizing flows across multiple scales to model the distribution
of normal images and detect anomalies.
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

from .loss import CsFlowLoss
from .torch_model import CsFlowModel

logger = logging.getLogger(__name__)

__all__ = ["Csflow"]


class Csflow(AnomalibModule):
    """CS-Flow Lightning Model for anomaly detection.

    CS-Flow uses normalizing flows across multiple scales to model the distribution
    of normal images. During inference, it assigns anomaly scores based on the
    likelihood of test samples under the learned distribution.

    Args:
        n_coupling_blocks (int, optional): Number of coupling blocks in the model.
            Defaults to ``4``.
        cross_conv_hidden_channels (int, optional): Number of hidden channels in
            the cross convolution layer. Defaults to ``1024``.
        clamp (int, optional): Clamping value for the affine coupling layers in
            the Glow model. Defaults to ``3``.
        num_channels (int, optional): Number of input image channels.
            Defaults to ``3`` for RGB images.
        pre_processor (PreProcessor | bool, optional): Preprocessing module or
            flag to enable default preprocessing. Defaults to ``True``.
        post_processor (PostProcessor | bool, optional): Post-processing module or
            flag to enable default post-processing. Defaults to ``True``.
        evaluator (Evaluator | bool, optional): Evaluation module or flag to
            enable default evaluation. Defaults to ``True``.
        visualizer (Visualizer | bool, optional): Visualization module or flag to
            enable default visualization. Defaults to ``True``.

    Raises:
        ValueError: If ``input_size`` is not provided during initialization.

    Example:
        >>> from anomalib.models.image.csflow import Csflow
        >>> model = Csflow(
        ...     n_coupling_blocks=4,
        ...     cross_conv_hidden_channels=1024,
        ...     clamp=3,
        ...     num_channels=3
        ... )
    """

    def __init__(
        self,
        cross_conv_hidden_channels: int = 1024,
        n_coupling_blocks: int = 4,
        clamp: int = 3,
        num_channels: int = 3,
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
        if self.input_size is None:
            msg = "CsFlow needs input size to build torch model."
            raise ValueError(msg)

        self.cross_conv_hidden_channels = cross_conv_hidden_channels
        self.n_coupling_blocks = n_coupling_blocks
        self.clamp = clamp
        self.num_channels = num_channels

        self.model = CsFlowModel(
            input_size=self.input_size,
            cross_conv_hidden_channels=self.cross_conv_hidden_channels,
            n_coupling_blocks=self.n_coupling_blocks,
            clamp=self.clamp,
            num_channels=self.num_channels,
        )
        self.model.feature_extractor.eval()
        self.loss = CsFlowLoss()

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a training step of CS-Flow model.

        Args:
            batch (Batch): Input batch containing images and targets
            *args: Additional positional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            STEP_OUTPUT: Dictionary containing the loss value

        Example:
            >>> batch = Batch(image=torch.randn(32, 3, 256, 256))
            >>> model = Csflow()
            >>> output = model.training_step(batch)
            >>> output["loss"]
            tensor(...)
        """
        del args, kwargs  # These variables are not used.

        z_dist, jacobians = self.model(batch.image)
        loss = self.loss(z_dist, jacobians)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a validation step of CS-Flow model.

        Args:
            batch (Batch): Input batch containing images and targets
            *args: Additional positional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            STEP_OUTPUT: Dictionary containing predictions including anomaly maps
                and scores

        Example:
            >>> batch = Batch(image=torch.randn(32, 3, 256, 256))
            >>> model = Csflow()
            >>> predictions = model.validation_step(batch)
        """
        del args, kwargs  # These variables are not used.

        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Get CS-Flow-specific trainer arguments.

        Returns:
            dict[str, Any]: Dictionary containing trainer arguments:
                - gradient_clip_val: Maximum gradient norm for clipping
                - num_sanity_val_steps: Number of validation steps to run before
                  training
        """
        return {"gradient_clip_val": 1, "num_sanity_val_steps": 0}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the Adam optimizer for CS-Flow.

        Returns:
            torch.optim.Optimizer: Configured Adam optimizer with specific
                hyperparameters

        Example:
            >>> model = Csflow()
            >>> optimizer = model.configure_optimizers()
        """
        return torch.optim.Adam(
            self.parameters(),
            lr=2e-4,
            eps=1e-04,
            weight_decay=1e-5,
            betas=(0.5, 0.9),
        )

    @property
    def learning_type(self) -> LearningType:
        """Get the learning type of the model.

        Returns:
            LearningType: The learning type, which is ONE_CLASS for CS-Flow
        """
        return LearningType.ONE_CLASS
