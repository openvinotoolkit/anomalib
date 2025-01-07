"""Deep Feature Modeling (DFM) for anomaly detection.

This module provides a PyTorch Lightning implementation of the DFM model for
anomaly detection. The model extracts deep features from images using a
pre-trained CNN backbone and fits a Gaussian model on these features to detect
anomalies.

Paper: https://arxiv.org/abs/1909.11786

Example:
    >>> from anomalib.models.image import Dfm
    >>> model = Dfm(
    ...     backbone="resnet50",
    ...     layer="layer3",
    ...     pre_trained=True
    ... )

Notes:
    The model uses a pre-trained backbone to extract features and fits a PCA
    transformation followed by a Gaussian model during training. No gradient
    updates are performed on the backbone.

See Also:
    :class:`anomalib.models.image.dfm.torch_model.DFMModel`:
        PyTorch implementation of the DFM model.
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
from anomalib.models.components import AnomalibModule, MemoryBankMixin
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .torch_model import DFMModel

logger = logging.getLogger(__name__)


class Dfm(MemoryBankMixin, AnomalibModule):
    """DFM Lightning Module.

    Args:
        backbone (str): Name of the backbone CNN network.
            Defaults to ``"resnet50"``.
        layer (str): Name of the layer to extract features from the backbone.
            Defaults to ``"layer3"``.
        pre_trained (bool, optional): Whether to use a pre-trained backbone.
            Defaults to ``True``.
        pooling_kernel_size (int, optional): Kernel size for pooling features.
            Defaults to ``4``.
        pca_level (float, optional): Ratio of variance to preserve in PCA.
            Must be between 0 and 1.
            Defaults to ``0.97``.
        score_type (str, optional): Type of anomaly score to compute.
            Options are ``"fre"`` (feature reconstruction error) or
            ``"nll"`` (negative log-likelihood).
            Defaults to ``"fre"``.
        pre_processor (PreProcessor | bool, optional): Pre-processor to use.
            If ``True``, uses the default pre-processor.
            If ``False``, no pre-processing is performed.
            Defaults to ``True``.
        post_processor (PostProcessor | bool, optional): Post-processor to use.
            If ``True``, uses the default post-processor.
            If ``False``, no post-processing is performed.
            Defaults to ``True``.
        evaluator (Evaluator | bool, optional): Evaluator to use.
            If ``True``, uses the default evaluator.
            If ``False``, no evaluation is performed.
            Defaults to ``True``.
        visualizer (Visualizer | bool, optional): Visualizer to use.
            If ``True``, uses the default visualizer.
            If ``False``, no visualization is performed.
            Defaults to ``True``.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        layer: str = "layer3",
        pre_trained: bool = True,
        pooling_kernel_size: int = 4,
        pca_level: float = 0.97,
        score_type: str = "fre",
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

        self.model: DFMModel = DFMModel(
            backbone=backbone,
            pre_trained=pre_trained,
            layer=layer,
            pooling_kernel_size=pooling_kernel_size,
            n_comps=pca_level,
            score_type=score_type,
        )
        self.embeddings: list[torch.Tensor] = []
        self.score_type = score_type

    @staticmethod
    def configure_optimizers() -> None:  # pylint: disable=arguments-differ
        """Configure optimizers for training.

        Returns:
            None: DFM doesn't require optimization.
        """
        return

    def training_step(self, batch: Batch, *args, **kwargs) -> None:
        """Extract features from the input batch during training.

        Args:
            batch (Batch): Input batch containing images.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            torch.Tensor: Dummy loss tensor for compatibility.
        """
        del args, kwargs  # These variables are not used.

        embedding = self.model.get_features(batch.image).squeeze()
        self.embeddings.append(embedding)

        # Return a dummy loss tensor
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def fit(self) -> None:
        """Fit the PCA transformation and Gaussian model to the embeddings.

        The method aggregates embeddings collected during training and fits
        both the PCA transformation and Gaussian model used for scoring.
        """
        logger.info("Aggregating the embedding extracted from the training set.")
        embeddings = torch.vstack(self.embeddings)

        logger.info("Fitting a PCA and a Gaussian model to dataset.")
        self.model.fit(embeddings)

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Compute predictions for the input batch during validation.

        Args:
            batch (Batch): Input batch containing images.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            STEP_OUTPUT: Dictionary containing anomaly scores and maps.
        """
        del args, kwargs  # These variables are not used.

        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Get DFM-specific trainer arguments.

        Returns:
            dict[str, Any]: Dictionary of trainer arguments:
                - ``gradient_clip_val`` (int): Disable gradient clipping
                - ``max_epochs`` (int): Train for one epoch only
                - ``num_sanity_val_steps`` (int): Skip validation sanity checks
        """
        return {"gradient_clip_val": 0, "max_epochs": 1, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Get the learning type of the model.

        Returns:
            LearningType: The model uses one-class learning.
        """
        return LearningType.ONE_CLASS
