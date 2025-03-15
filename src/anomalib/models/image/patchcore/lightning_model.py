"""PatchCore: Towards Total Recall in Industrial Anomaly Detection.

This module implements the PatchCore model for anomaly detection using a memory bank
of patch features extracted from a pretrained CNN backbone. The model stores
representative patch features from normal training images and detects anomalies by
comparing test image patches against this memory bank.

The model uses a nearest neighbor search to find the most similar patches in the
memory bank and computes anomaly scores based on these distances. It achieves high
performance while maintaining interpretability through localization maps.

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import Patchcore
    >>> from anomalib.engine import Engine

    >>> # Initialize model and data
    >>> datamodule = MVTecAD()
    >>> model = Patchcore(
    ...     backbone="wide_resnet50_2",
    ...     layers=["layer2", "layer3"],
    ...     coreset_sampling_ratio=0.1
    ... )

    >>> # Train using the Engine
    >>> engine = Engine()
    >>> engine.fit(model=model, datamodule=datamodule)

    >>> # Get predictions
    >>> predictions = engine.predict(model=model, datamodule=datamodule)

Paper: https://arxiv.org/abs/2106.08265

See Also:
    - :class:`anomalib.models.image.patchcore.torch_model.PatchcoreModel`:
        PyTorch implementation of the PatchCore model architecture
    - :class:`anomalib.models.image.patchcore.anomaly_map.AnomalyMapGenerator`:
        Anomaly map generation for PatchCore using nearest neighbor search
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence
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

from .torch_model import PatchcoreModel

logger = logging.getLogger(__name__)


class Patchcore(MemoryBankMixin, AnomalibModule):
    """PatchCore Lightning Module for anomaly detection.

    This class implements the PatchCore algorithm which uses a memory bank of patch
    features for anomaly detection. Features are extracted from a pretrained CNN
    backbone and stored in a memory bank. Anomalies are detected by comparing test
    image patches with the stored features using nearest neighbor search.

    The model works in two phases:
    1. Training: Extract and store patch features from normal training images
    2. Inference: Compare test image patches against stored features to detect
       anomalies

    Args:
        backbone (str): Name of the backbone CNN network.
            Defaults to ``"wide_resnet50_2"``.
        layers (Sequence[str]): Names of layers to extract features from.
            Defaults to ``("layer2", "layer3")``.
        pre_trained (bool, optional): Whether to use pre-trained backbone weights.
            Defaults to ``True``.
        coreset_sampling_ratio (float, optional): Ratio for coreset sampling to
            subsample embeddings. Defaults to ``0.1``.
        num_neighbors (int, optional): Number of nearest neighbors to use.
            Defaults to ``9``.
        pre_processor (PreProcessor | bool, optional): Pre-processor instance or
            bool flag. Defaults to ``True``.
        post_processor (PostProcessor | bool, optional): Post-processor instance or
            bool flag. Defaults to ``True``.
        evaluator (Evaluator | bool, optional): Evaluator instance or bool flag.
            Defaults to ``True``.
        visualizer (Visualizer | bool, optional): Visualizer instance or bool flag.
            Defaults to ``True``.

    Example:
        >>> from anomalib.data import MVTecAD
        >>> from anomalib.models import Patchcore
        >>> from anomalib.engine import Engine

        >>> # Initialize model and data
        >>> datamodule = MVTecAD()
        >>> model = Patchcore(
        ...     backbone="wide_resnet50_2",
        ...     layers=["layer2", "layer3"],
        ...     coreset_sampling_ratio=0.1
        ... )

        >>> # Train using the Engine
        >>> engine = Engine()
        >>> engine.fit(model=model, datamodule=datamodule)

        >>> # Get predictions
        >>> predictions = engine.predict(model=model, datamodule=datamodule)

    Notes:
        The model requires no optimization/backpropagation as it uses a pretrained
        backbone and nearest neighbor search.

    See Also:
        - :class:`anomalib.models.components.AnomalibModule`:
            Base class for all anomaly detection models
        - :class:`anomalib.models.components.MemoryBankMixin`:
            Mixin class for models using feature memory banks
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer2", "layer3"),
        pre_trained: bool = True,
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
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

        self.model: PatchcoreModel = PatchcoreModel(
            backbone=backbone,
            pre_trained=pre_trained,
            layers=layers,
            num_neighbors=num_neighbors,
        )
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.embeddings: list[torch.Tensor] = []

    @classmethod
    def configure_pre_processor(
        cls,
        image_size: tuple[int, int] | None = None,
        center_crop_size: tuple[int, int] | None = None,
    ) -> PreProcessor:
        """Configure the default pre-processor for PatchCore.

        The pre-processor performs the following steps:
        1. Resize image to specified size
        2. Center crop to maintain aspect ratio
        3. Normalize using ImageNet mean and std

        Args:
            image_size (tuple[int, int] | None, optional): Target size for
                resizing. Defaults to ``(256, 256)``.
            center_crop_size (tuple[int, int] | None, optional): Size for center
                cropping. If ``None``, scales proportionally to ``image_size``.
                Defaults to ``None``.

        Returns:
            PreProcessor: Configured pre-processor instance.

        Example:
            >>> pre_processor = Patchcore.configure_pre_processor(
            ...     image_size=(256, 256)
            ... )
            >>> transformed_image = pre_processor(image)
        """
        image_size = image_size or (256, 256)
        if center_crop_size is None:
            # scale center crop size proportional to image size
            height, width = image_size
            center_crop_size = (int(height * (224 / 256)), int(width * (224 / 256)))

        transform = Compose([
            Resize(image_size, antialias=True),
            CenterCrop(center_crop_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return PreProcessor(transform=transform)

    @staticmethod
    def configure_optimizers() -> None:
        """Configure optimizers.

        Returns:
            None: PatchCore requires no optimization.
        """
        return

    def training_step(self, batch: Batch, *args, **kwargs) -> None:
        """Generate feature embedding of the batch.

        Args:
            batch (Batch): Input batch containing image and metadata
            *args: Additional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            torch.Tensor: Dummy loss tensor for Lightning compatibility

        Note:
            The method stores embeddings in ``self.embeddings`` for later use in
            ``fit()``.
        """
        del args, kwargs  # These variables are not used.

        embedding = self.model(batch.image)
        self.embeddings.append(embedding)
        # Return a dummy loss tensor
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def fit(self) -> None:
        """Apply subsampling to the embedding collected from the training set.

        This method:
        1. Aggregates embeddings from all training batches
        2. Applies coreset subsampling to reduce memory requirements
        """
        logger.info("Aggregating the embedding extracted from the training set.")
        embeddings = torch.vstack(self.embeddings)

        logger.info("Applying core-set subsampling to get the embedding.")
        self.model.subsample_embedding(embeddings, self.coreset_sampling_ratio)

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Generate predictions for a batch of images.

        Args:
            batch (Batch): Input batch containing images and metadata
            *args: Additional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            STEP_OUTPUT: Batch with added predictions

        Note:
            Predictions include anomaly maps and scores computed using nearest
            neighbor search.
        """
        # These variables are not used.
        del args, kwargs

        # Get anomaly maps and predicted scores from the model.
        predictions = self.model(batch.image)

        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Get default trainer arguments for PatchCore.

        Returns:
            dict[str, Any]: Trainer arguments
                - ``gradient_clip_val``: ``0`` (no gradient clipping needed)
                - ``max_epochs``: ``1`` (single pass through training data)
                - ``num_sanity_val_steps``: ``0`` (skip validation sanity checks)
        """
        return {"gradient_clip_val": 0, "max_epochs": 1, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Get the learning type.

        Returns:
            LearningType: Always ``LearningType.ONE_CLASS`` as PatchCore only
                trains on normal samples
        """
        return LearningType.ONE_CLASS

    @staticmethod
    def configure_post_processor() -> PostProcessor:
        """Configure the default post-processor.

        Returns:
            PostProcessor: Post-processor for one-class models that
                converts raw scores to anomaly predictions
        """
        return PostProcessor()
