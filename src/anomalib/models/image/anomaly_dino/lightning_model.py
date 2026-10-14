# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2.

This module implements AnomalyDINO. A memory-bank model for anomaly detection
that utilizes DINOv2-Small as its backbone. At inference time it uses kNN
to search for anomalous patches. The image anomaly score is dependent on the worst
99th percentile of the pixel-wise anomaly score.

The model has optional masking to remove noisy background components,
also optionally can use greedy coreset-subsampling if needed.

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models.image.anomaly_dino.lightning_model import AnomalyDINO
    >>> from anomalib.engine import Engine

    >>> MVTEC_CATEGORIES = [
    ...     "hazelnut", "grid", "carpet", "bottle", "cable", "capsule", "leather",
    ...     "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"
    ... ]
    >>> MASKED_CATEGORIES = ["capsule", "hazelnut", "pill", "screw", "toothbrush"]

    >>> for category in MVTEC_CATEGORIES:
    ...     mask = category in MASKED_CATEGORIES
    ...     print(f"--- Running category: {category} | masking={mask} ---")

    ...     # Initialize data module
    ...     datamodule = MVTecAD(category=category)

    ...     # Initialize model
    ...     model = AnomalyDINO(
    ...         num_neighbours=1,
    ...         encoder_name="vit_small_patch14_dinov2",
    ...         masking=mask,
    ...         coreset_subsampling=False,
    ...     )

    ...     # Train and test
    ...     engine = Engine()
    ...     engine.fit(model=model, datamodule=datamodule)
    ...     engine.test(datamodule=datamodule)
    >>> print("All categories processed.")
"""

import logging
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torchvision.transforms.v2 import Compose, InterpolationMode, Normalize, Resize

from anomalib import LearningType, PrecisionType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule, MemoryBankMixin
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .torch_model import AnomalyDINOModel

logger = logging.getLogger(__name__)


class AnomalyDINO(MemoryBankMixin, AnomalibModule):
    """AnomalyDINO Lightning Module for anomaly detection.

    This class implements the AnomalyDINO algorithm, which leverages self-supervised
    DINO (self-distillation with no labels) vision transformer (ViT) encoders for
    feature extraction in anomaly detection tasks. Similar to PatchCore, it uses a
    memory bank of patch embeddings and performs nearest neighbor search to identify
    anomalous regions in test images.

    The model operates in two phases:
    1. **Training**: Extracts and stores patch embeddings from normal training images.
    2. **Inference**: Compares test image patch embeddings with the memory bank
       to identify anomalies based on distance metrics.

    Args:
        num_neighbours (int, optional): Number of nearest neighbors to use for
            anomaly scoring. Defaults to ``1``.
        encoder_name (str, optional): Name of the pretrained DINO encoder to use.
            Defaults to ``"vit_small_patch14_dinov2"``.
        masking (bool, optional): Whether to apply masking during feature extraction
            to simulate occlusions or missing patches. Defaults to ``False``.
        coreset_subsampling (bool, optional): Whether to apply coreset subsampling
            to reduce the size of the memory bank. Defaults to ``False``.
        sampling_ratio (float, optional): If coreset subsampling, by what ratio
            should we subsample. Defaults to ``0.1``
        precision (str | PrecisionType, optional): Precision type for model computations.
            Can be either a string (``"float32"``, ``"float16"``) or a :class:`PrecisionType` enum value.
            Defaults to ``PrecisionType.FLOAT32``.
        pre_processor (PreProcessor | bool, optional): Pre-processor instance or
            bool flag to enable default preprocessing. Defaults to ``True``.
        post_processor (PostProcessor | bool, optional): Post-processor instance or
            bool flag to enable default postprocessing. Defaults to ``True``.
        evaluator (Evaluator | bool, optional): Evaluator instance or bool flag for
            performance computation. Defaults to ``True``.
        visualizer (Visualizer | bool, optional): Visualizer instance or bool flag
            to enable visualization. Defaults to ``True``.

    Example:
        >>> from anomalib.data import MVTecAD
        >>> from anomalib.models.image.anomaly_dino.lightning_model import AnomalyDINO
        >>> from anomalib.engine import Engine

        >>> MVTEC_CATEGORIES = [
        ...     "hazelnut", "grid", "carpet", "bottle", "cable", "capsule", "leather",
        ...     "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"
        ... ]
        >>> MASKED_CATEGORIES = ["capsule", "hazelnut", "pill", "screw", "toothbrush"]

        >>> for category in MVTEC_CATEGORIES:
        ...     mask = category in MASKED_CATEGORIES
        ...     print(f"--- Running category: {category} | masking={mask} ---")

        ...     # Initialize data module
        ...     datamodule = MVTecAD(category=category)

        ...     # Initialize model
        ...     model = AnomalyDINO(
        ...         num_neighbours=1,
        ...         encoder_name="vit_small_patch14_dinov2",
        ...         masking=mask,
        ...         coreset_subsampling=False,
        ...     )

        ...     # Train and test
        ...     engine = Engine()
        ...     engine.fit(model=model, datamodule=datamodule)
        ...     engine.test(datamodule=datamodule)

        >>> print("All categories processed.")

    Notes:
        - The model does not require backpropagation or optimization, as it relies
          on pretrained transformer embeddings and similarity search.
        - Works best when trained exclusively on normal (non-anomalous) samples.

    See Also:
        - :class:`anomalib.models.components.AnomalibModule`:
            Base class for all anomaly detection models
        - :class:`anomalib.models.components.MemoryBankMixin`:
            Mixin class for models using memory bank embeddings
    """

    def __init__(
        self,
        num_neighbours: int = 1,
        encoder_name: str = "vit_small_patch14_dinov2",
        masking: bool = False,
        coreset_subsampling: bool = False,
        sampling_ratio: float = 0.1,
        precision: str | PrecisionType = PrecisionType.FLOAT32,
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
        self.model: AnomalyDINOModel = AnomalyDINOModel(
            num_neighbours=num_neighbours,
            encoder_name=encoder_name,
            masking=masking,
            coreset_subsampling=coreset_subsampling,
            sampling_ratio=sampling_ratio,
        )

        if isinstance(precision, str):
            precision = PrecisionType(precision.lower())

        if precision == PrecisionType.FLOAT16:
            self.model = self.model.half()
        elif precision == PrecisionType.FLOAT32:
            self.model = self.model.float()
        else:
            msg = (
                f"Unsupported precision type: {precision}. "
                f"Supported types are: {PrecisionType.FLOAT16}, {PrecisionType.FLOAT32}."
            )
            raise ValueError(msg)

    @classmethod
    def configure_pre_processor(
        cls,
        image_size: tuple[int, int] | int | None = None,
    ) -> PreProcessor:
        """Configure the default pre-processor for AnomalyDINO.

        Args:
            image_size (tuple[int, int] | int | None, optional): Target size for resizing
                input images. Defaults to ``(252, 252)``. Note if int, keeps aspect ratio and resizes shortest side.

        Returns:
            PreProcessor: Configured pre-processor instance.

        Example:
            >>> pre_processor = AnomalyDINO.configure_pre_processor(
            ...     image_size=(252, 252)
            ... )
            >>> transformed_image = pre_processor(image)
        """
        image_size = image_size or (252, 252)
        transform = Compose([
            Resize(image_size, antialias=True, interpolation=InterpolationMode.BICUBIC),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return PreProcessor(transform=transform)

    @staticmethod
    def configure_optimizers() -> None:
        """Configure optimizers.

        Returns:
            None: AnomalyDINO does not require optimization or gradient updates.
        """
        return

    def training_step(self, batch: Batch, *args, **kwargs) -> None:
        """Extract feature embeddings from training images.

        Args:
            batch (Batch): Input batch containing images and metadata.
            *args: Additional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            torch.Tensor: Dummy loss tensor for Lightning compatibility.

        Note:
            The extracted embeddings are stored in the models memory bank for
            later use during the coreset sampling or inference phase.
        """
        del args, kwargs  # These variables are not used.
        _ = self.model(batch.image)
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def fit(self) -> None:
        """Optional fitting step.

        This method is a placeholder for potential post-training operations
        such as coreset subsampling or feature normalization. The model
        handles fitting (if-needed).
        """
        self.model.fit()

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Generate anomaly predictions for a validation batch.

        Args:
            batch (Batch): Input batch containing images and metadata.
            *args: Additional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            STEP_OUTPUT: Batch with added predictions including anomaly maps and
                scores computed using nearest neighbor search.
        """
        del args, kwargs
        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Default PyTorch Lightning trainer arguments for AnomalyDINO.

        Returns:
            dict[str, Any]: Trainer configuration with:
                - ``gradient_clip_val``: ``0`` (no gradient clipping)
                - ``max_epochs``: ``1`` (single pass over training data)
                - ``num_sanity_val_steps``: ``0`` (skip validation sanity checks)
                - ``devices``: ``1`` (single GPU supported)
        """
        return {"gradient_clip_val": 0, "max_epochs": 1, "num_sanity_val_steps": 0, "devices": 1}

    @property
    def learning_type(self) -> LearningType:
        """Get the learning type for AnomalyDINO.

        Returns:
            LearningType: Always ``LearningType.ONE_CLASS`` since the model is
            trained only on normal samples.
        """
        return LearningType.ONE_CLASS

    @staticmethod
    def configure_post_processor() -> PostProcessor:
        """Configure the default post-processor.

        Returns:
            PostProcessor: Post-processor that converts raw model scores into
                interpretable anomaly predictions and maps.
        """
        return PostProcessor()
