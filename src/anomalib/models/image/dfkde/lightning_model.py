"""DFKDE: Deep Feature Kernel Density Estimation.

This module provides a PyTorch Lightning implementation of the DFKDE model for
anomaly detection. The model extracts deep features from images using a
pre-trained CNN backbone and fits a kernel density estimation on these features
to model the distribution of normal samples.

Example:
    >>> from anomalib.models.image import Dfkde
    >>> model = Dfkde(
    ...     backbone="resnet18",
    ...     layers=("layer4",),
    ...     pre_trained=True
    ... )

Notes:
    The model uses a pre-trained backbone to extract features and fits a KDE
    classifier on the embeddings during training. No gradient updates are
    performed on the backbone.

See Also:
    :class:`anomalib.models.image.dfkde.torch_model.DfkdeModel`:
        PyTorch implementation of the DFKDE model.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import AUROC, Evaluator, F1Score
from anomalib.models.components import AnomalibModule, MemoryBankMixin
from anomalib.models.components.classification import FeatureScalingMethod
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .torch_model import DfkdeModel

logger = logging.getLogger(__name__)


class Dfkde(MemoryBankMixin, AnomalibModule):
    """DFKDE Lightning Module.

    Args:
        backbone (str): Name of the backbone CNN to use for feature extraction.
            Defaults to ``"resnet18"``.
        layers (Sequence[str]): Layers from which to extract features.
            Defaults to ``("layer4",)``.
        pre_trained (bool): Whether to use pre-trained weights.
            Defaults to ``True``.
        n_pca_components (int): Number of principal components for dimensionality
            reduction. Defaults to ``16``.
        feature_scaling_method (FeatureScalingMethod): Method to scale features.
            Defaults to ``FeatureScalingMethod.SCALE``.
        max_training_points (int): Maximum number of points to use for KDE
            fitting. Defaults to ``40000``.
        pre_processor (PreProcessor | bool): Pre-processor object or flag.
            Defaults to ``True``.
        post_processor (PostProcessor | bool): Post-processor object or flag.
            Defaults to ``True``.
        evaluator (Evaluator | bool): Evaluator object or flag.
            Defaults to ``True``.
        visualizer (Visualizer | bool): Visualizer object or flag.
            Defaults to ``True``.

    Example:
        >>> from anomalib.models.image import Dfkde
        >>> from anomalib.models.components.classification import (
        ...     FeatureScalingMethod
        ... )
        >>> model = Dfkde(
        ...     backbone="resnet18",
        ...     layers=("layer4",),
        ...     feature_scaling_method=FeatureScalingMethod.SCALE
        ... )
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: Sequence[str] = ("layer4",),
        pre_trained: bool = True,
        n_pca_components: int = 16,
        feature_scaling_method: FeatureScalingMethod = FeatureScalingMethod.SCALE,
        max_training_points: int = 40000,
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

        self.model = DfkdeModel(
            layers=layers,
            backbone=backbone,
            pre_trained=pre_trained,
            n_pca_components=n_pca_components,
            feature_scaling_method=feature_scaling_method,
            max_training_points=max_training_points,
        )

        self.embeddings: list[torch.Tensor] = []

    @staticmethod
    def configure_optimizers() -> None:  # pylint: disable=arguments-differ
        """DFKDE doesn't require optimization, therefore returns no optimizers."""
        return

    def training_step(self, batch: Batch, *args, **kwargs) -> None:
        """Extract features from the CNN for each training batch.

        Args:
            batch (Batch): Input batch containing images and metadata.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: Dummy tensor for Lightning compatibility.
        """
        del args, kwargs  # These variables are not used.

        embedding = self.model(batch.image)
        self.embeddings.append(embedding)

        # Return a dummy loss tensor
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def fit(self) -> None:
        """Fit KDE model to collected embeddings from the training set."""
        embeddings = torch.vstack(self.embeddings)

        logger.info("Fitting a KDE model to the embedding collected from the training set.")
        self.model.classifier.fit(embeddings)

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform validation by computing anomaly scores.

        Args:
            batch (Batch): Input batch containing images and metadata.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            STEP_OUTPUT: Dictionary containing predictions and batch info.
        """
        del args, kwargs  # These variables are not used.

        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Get DFKDE-specific trainer arguments.

        Returns:
            dict[str, Any]: Dictionary of trainer arguments.
        """
        return {"gradient_clip_val": 0, "max_epochs": 1, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Get the learning type.

        Returns:
            LearningType: Learning type of the model (ONE_CLASS).
        """
        return LearningType.ONE_CLASS

    @staticmethod
    def configure_evaluator() -> Evaluator:
        """Configure the default evaluator for DFKDE.

        Returns:
            Evaluator: Evaluator object with image-level AUROC and F1 metrics.
        """
        image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
        image_f1score = F1Score(fields=["pred_label", "gt_label"], prefix="image_")
        test_metrics = [image_auroc, image_f1score]
        return Evaluator(test_metrics=test_metrics)
