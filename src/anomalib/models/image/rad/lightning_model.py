# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RAD: Retrieval-based Anomaly Detection.

This module implements the RAD Lightning model for training-free anomaly
detection using multi-layer retrieval with position-aware patch matching.

Paper: https://arxiv.org/abs/2601.22763

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import Rad
    >>> from anomalib.engine import Engine

    >>> datamodule = MVTecAD()
    >>> model = Rad()

    >>> engine = Engine()
    >>> engine.fit(model=model, datamodule=datamodule)
    >>> predictions = engine.predict(model=model, datamodule=datamodule)
"""

import logging
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule, MemoryBankMixin
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .torch_model import RadModel

logger = logging.getLogger(__name__)


class Rad(MemoryBankMixin, AnomalibModule):
    """RAD Lightning Module for anomaly detection.

    Implements Retrieval-based Anomaly Detection (RAD), a training-free framework
    that stores anomaly-free features in a multi-layer memory bank and detects
    anomalies through image-level retrieval followed by position-aware patch
    matching.

    Args:
        backbone (str): Name of the ViT backbone from timm.
            Defaults to ``"vit_base_patch16_dinov3"``.
        layers (list[int] | None): Block indices (0-based) to extract features from.
            Defaults to ``[3, 6, 9, 11]``.
        pre_trained (bool): Whether to use pre-trained backbone weights.
            Defaults to ``True``.
        k_image (int): Number of nearest-neighbor training images retrieved per test image.
            The paper uses ``150`` for MVTec-AD, ``900`` for VisA and Real-IAD, and ``48``
            for 3D-ADAM. Defaults to ``150``.
        use_positional_bank (bool): Enable position-aware patch matching.
            Defaults to ``True``.
        pos_radius (int): Spatial neighborhood radius (in patch units). The paper uses ``1``
            for MVTec-AD and 3D-ADAM, ``2`` for VisA, and ``0`` for Real-IAD.
            Defaults to ``1``.
        max_ratio (float): Fraction of highest anomaly pixels pooled for
            image-level score. ``0`` means use max. Defaults to ``0.01``.
        layer_weights (list[float] | None): Per-layer score fusion weights.
            ``None`` means uniform. Defaults to ``None``.
        bank_dtype (torch.dtype | str | None): Storage dtype of the memory bank. ``None``
            keeps the backbone dtype. ``torch.float16`` halves bank memory and checkpoint
            size and speeds up scoring, at the cost of a small perturbation of the scores.
            Defaults to ``None``.
        pre_processor (PreProcessor | bool): Pre-processor instance or flag.
            Defaults to ``True``.
        post_processor (PostProcessor | bool): Post-processor instance or flag.
            Defaults to ``True``.
        evaluator (Evaluator | bool): Evaluator instance or flag.
            Defaults to ``True``.
        visualizer (Visualizer | bool): Visualizer instance or flag.
            Defaults to ``True``.

    Note:
        RAD is training-free: fitting only fills the memory bank. The bank is stored in the
        model state, so checkpoints grow with the size of the anomaly-free training set.

    Example:
        >>> from anomalib.data import MVTecAD
        >>> from anomalib.models import Rad
        >>> from anomalib.engine import Engine

        >>> datamodule = MVTecAD()
        >>> model = Rad(
        ...     backbone="vit_base_patch16_dinov3",
        ...     layers=[3, 6, 9, 11],
        ...     k_image=150,
        ... )

        >>> engine = Engine()
        >>> engine.fit(model=model, datamodule=datamodule)
    """

    def __init__(
        self,
        backbone: str = "vit_base_patch16_dinov3",
        layers: list[int] | None = None,
        pre_trained: bool = True,
        k_image: int = 150,
        use_positional_bank: bool = True,
        pos_radius: int = 1,
        max_ratio: float = 0.01,
        layer_weights: list[float] | None = None,
        bank_dtype: torch.dtype | str | None = None,
        pre_processor: nn.Module | bool = True,
        post_processor: nn.Module | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        self.model = RadModel(
            backbone=backbone,
            pre_trained=pre_trained,
            layers=layers,
            k_image=k_image,
            use_positional_bank=use_positional_bank,
            pos_radius=pos_radius,
            max_ratio=max_ratio,
            layer_weights=layer_weights,
            bank_dtype=bank_dtype,
        )

    @classmethod
    def configure_pre_processor(
        cls,
        image_size: tuple[int, int] | None = None,
        center_crop_size: tuple[int, int] | None = None,
    ) -> PreProcessor:
        """Configure the default pre-processor for RAD.

        Args:
            image_size (tuple[int, int] | None): Target resize dimensions.
                Defaults to ``(512, 512)``.
            center_crop_size (tuple[int, int] | None): Center crop dimensions.
                Defaults to ``(448, 448)``.

        Returns:
            PreProcessor: Configured pre-processor instance.

        Example:
            >>> pre_processor = Rad.configure_pre_processor()
        """
        image_size = image_size or (512, 512)
        center_crop_size = center_crop_size or (448, 448)

        transform = Compose(
            [
                Resize(image_size, antialias=True),
                CenterCrop(center_crop_size),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        )

        return PreProcessor(transform=transform)

    @staticmethod
    def configure_optimizers() -> None:
        """Configure optimizers.

        Returns:
            None: RAD requires no optimization.
        """
        return

    def training_step(self, batch: Batch, *args, **kwargs) -> torch.Tensor:
        """Extract and store features from training batch.

        Args:
            batch (Batch): Input batch containing images.
            *args: Additional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            torch.Tensor: Dummy loss tensor for Lightning compatibility.
        """
        del args, kwargs
        _ = self.model(batch.image)
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def fit(self) -> None:
        """Build memory bank from accumulated training features."""
        logger.info("Building RAD multi-layer memory bank.")
        self.model.build_memory_bank()

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Generate predictions for a batch of images.

        Args:
            batch (Batch): Input batch containing images.
            *args: Additional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            STEP_OUTPUT: Batch with predictions.
        """
        del args, kwargs
        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Default trainer arguments.

        Returns:
            dict[str, Any]: Trainer arguments for single-epoch feature extraction.
        """
        return {
            "gradient_clip_val": 0,
            "max_epochs": 1,
            "num_sanity_val_steps": 0,
            "devices": 1,
        }

    @property
    def learning_type(self) -> LearningType:
        """Learning type of this model.

        Returns:
            LearningType: ``LearningType.ONE_CLASS``.
        """
        return LearningType.ONE_CLASS

    @staticmethod
    def configure_post_processor() -> PostProcessor:
        """Configure the default post-processor.

        Returns:
            PostProcessor: Default one-class post-processor.
        """
        return PostProcessor()
