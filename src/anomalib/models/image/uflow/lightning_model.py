"""U-Flow: A U-shaped Normalizing Flow for Anomaly Detection with Unsupervised Threshold.

This module implements the U-Flow model for anomaly detection as described in
 <https://arxiv.org/pdf/2211.12353.pdf>`_. The model consists
of:

- A U-shaped normalizing flow architecture for density estimation
- Multi-scale feature extraction using pre-trained backbones
- Unsupervised threshold estimation based on the learned density
- Anomaly detection by comparing likelihoods to the threshold

Example:
    >>> from anomalib.models.image import Uflow
    >>> from anomalib.engine import Engine
    >>> from anomalib.data import MVTecAD
    >>> datamodule = MVTecAD()
    >>> model = Uflow()
    >>> engine = Engine(model=model, datamodule=datamodule)
    >>> engine.fit()  # doctest: +SKIP
    >>> predictions = engine.predict()  # doctest: +SKIP

See Also:
    - :class:`UflowModel`: PyTorch implementation of the model architecture
    - :class:`UFlowLoss`: Loss function for training
    - :class:`AnomalyMapGenerator`: Anomaly map generation from features
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

import torch
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.optim.lr_scheduler import LRScheduler
from torchvision.transforms.v2 import Compose, Normalize, Resize

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .loss import UFlowLoss
from .torch_model import UflowModel

logger = logging.getLogger(__name__)

__all__ = ["Uflow"]


class Uflow(AnomalibModule):
    """Lightning implementation of the U-Flow model.

    This class implements the U-Flow model for anomaly detection as described in
    Rudolph et al., 2022. The model consists of:

    - A U-shaped normalizing flow architecture for density estimation
    - Multi-scale feature extraction using pre-trained backbones
    - Unsupervised threshold estimation based on the learned density
    - Anomaly detection by comparing likelihoods to the threshold

    Args:
        backbone (str, optional): Name of the backbone feature extractor. Must be
            one of ``["mcait", "resnet18", "wide_resnet50_2"]``. Defaults to
            ``"mcait"``.
        flow_steps (int, optional): Number of normalizing flow steps. Defaults to
            ``4``.
        affine_clamp (float, optional): Clamping value for affine coupling
            layers. Defaults to ``2.0``.
        affine_subnet_channels_ratio (float, optional): Channel ratio for affine
            coupling subnet. Defaults to ``1.0``.
        permute_soft (bool, optional): Whether to use soft permutation. Defaults
            to ``False``.
        pre_processor (PreProcessor | bool, optional): Pre-processor for input
            data. If ``True``, uses default pre-processor. Defaults to ``True``.
        post_processor (PostProcessor | bool, optional): Post-processor for model
            outputs. If ``True``, uses default post-processor. Defaults to
            ``True``.
        evaluator (Evaluator | bool, optional): Evaluator for model performance.
            If ``True``, uses default evaluator. Defaults to ``True``.
        visualizer (Visualizer | bool, optional): Visualizer for model outputs.
            If ``True``, uses default visualizer. Defaults to ``True``.

    Example:
        >>> from anomalib.models.image import Uflow
        >>> from anomalib.engine import Engine
        >>> from anomalib.data import MVTecAD
        >>> datamodule = MVTecAD()
        >>> model = Uflow(backbone="resnet18")
        >>> engine = Engine(model=model, datamodule=datamodule)
        >>> engine.fit()  # doctest: +SKIP
        >>> predictions = engine.predict()  # doctest: +SKIP

    Raises:
        ValueError: If ``input_size`` is not provided during initialization.

    See Also:
        - :class:`UflowModel`: PyTorch implementation of the model architecture
        - :class:`UFlowLoss`: Loss function for training
        - :class:`AnomalyMapGenerator`: Anomaly map generation from features
    """

    def __init__(
        self,
        backbone: str = "mcait",
        flow_steps: int = 4,
        affine_clamp: float = 2.0,
        affine_subnet_channels_ratio: float = 1.0,
        permute_soft: bool = False,
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        """Initialize U-Flow model.

        Args:
            backbone (str, optional): Name of the backbone feature extractor.
                Must be one of ``["mcait", "resnet18", "wide_resnet50_2"]``.
                Defaults to ``"mcait"``.
            flow_steps (int, optional): Number of normalizing flow steps.
                Defaults to ``4``.
            affine_clamp (float, optional): Clamping value for affine coupling
                layers. Defaults to ``2.0``.
            affine_subnet_channels_ratio (float, optional): Channel ratio for
                affine coupling subnet. Defaults to ``1.0``.
            permute_soft (bool, optional): Whether to use soft permutation.
                Defaults to ``False``.
            pre_processor (PreProcessor | bool, optional): Pre-processor for
                input data. If ``True``, uses default pre-processor. Defaults to
                ``True``.
            post_processor (PostProcessor | bool, optional): Post-processor for
                model outputs. If ``True``, uses default post-processor.
                Defaults to ``True``.
            evaluator (Evaluator | bool, optional): Evaluator for model
                performance. If ``True``, uses default evaluator. Defaults to
                ``True``.
            visualizer (Visualizer | bool, optional): Visualizer for model
                outputs. If ``True``, uses default visualizer. Defaults to
                ``True``.

        Raises:
            ValueError: If ``input_size`` is not provided during initialization.
        """
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
        if self.input_size is None:
            msg = "Input size is required for UFlow model."
            raise ValueError(msg)

        self.backbone = backbone
        self.flow_steps = flow_steps
        self.affine_clamp = affine_clamp
        self.affine_subnet_channels_ratio = affine_subnet_channels_ratio
        self.permute_soft = permute_soft

        self.model = UflowModel(
            input_size=self.input_size,
            backbone=self.backbone,
            flow_steps=self.flow_steps,
            affine_clamp=self.affine_clamp,
            affine_subnet_channels_ratio=self.affine_subnet_channels_ratio,
            permute_soft=self.permute_soft,
        )
        self.loss = UFlowLoss()

    @classmethod
    def configure_pre_processor(cls, image_size: tuple[int, int] | None = None) -> PreProcessor:
        """Configure default pre-processor for U-Flow model.

        Args:
            image_size (tuple[int, int] | None, optional): Input image size.
                Not used as U-Flow has fixed input size. Defaults to ``None``.

        Returns:
            PreProcessor: Default pre-processor with resizing and normalization.

        Note:
            The input image size is fixed to 448x448 for U-Flow regardless of
            the provided ``image_size``.
        """
        if image_size is not None:
            logger.warning("Image size is not used in UFlow. The input image size is determined by the model.")
        transform = Compose([
            Resize((448, 448), antialias=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return PreProcessor(transform=transform)

    def configure_optimizers(self) -> tuple[list[LightningOptimizer], list[LRScheduler]]:
        """Configure optimizers and learning rate schedulers.

        Returns:
            tuple[list[LightningOptimizer], list[LRScheduler]]: Tuple containing:
                - List of optimizers (Adam with initial lr=1e-3)
                - List of schedulers (LinearLR reducing to 0.4 over 25000 steps)
        """
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

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:  # noqa: ARG002 | unused arguments
        """Perform a training step.

        Args:
            batch (Batch): Input batch containing images
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments

        Returns:
            STEP_OUTPUT: Dictionary containing the loss value
        """
        z, ljd = self.model(batch.image)
        loss = self.loss(z, ljd)
        self.log_dict({"loss": loss}, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:  # noqa: ARG002 | unused arguments
        """Perform a validation step.

        Args:
            batch (Batch): Input batch containing images
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments

        Returns:
            STEP_OUTPUT: Batch updated with model predictions
        """
        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Get trainer arguments for U-Flow.

        Returns:
            dict[str, Any]: Dictionary containing trainer arguments
        """
        return {"num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Get the learning type of the model.

        Returns:
            LearningType: Learning type (ONE_CLASS for U-Flow)
        """
        return LearningType.ONE_CLASS
