"""FUVAS: Few-shot Unsupervised Video Anomaly Segmentation via Low-Rank Factorization of Spatio-Temporal Features.

This module provides a PyTorch Lightning implementation of the FUVAS model for
video anomaly detection and segmentation. The model extracts deep features from video clips
using a pre-trained 3D CNN/transformer backbone and fits a PCA-based reconstruction model
to detect anomalies.

Paper: https://ieeexplore.ieee.org/abstract/document/10887597

Example:
    >>> from anomalib.models.video import fuvas
    >>> model = fuvas(
    ...     backbone="x3d_s",
    ...     layer="blocks.4",
    ...     pre_trained=True
    ... )

Notes:
    The model uses a pre-trained backbone to extract features and fits a PCA
    transformation during training. No gradient updates are performed on the backbone.
    Anomaly detection is based on feature reconstruction error.

See Also:
    :class:`anomalib.models.video.fuvas.torch_model.FUVASModel`:
        PyTorch implementation of the FUVAS model.
"""

# Copyright (C) 2025 Intel Corporation
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
from anomalib import TaskType

from .torch_model import FUVASModel

logger = logging.getLogger(__name__)


class Fuvas(MemoryBankMixin, AnomalibModule):
    """FUVAS Lightning Module.

    Args:
        backbone (str): Name of the backbone 3D CNN/transformer network.
            Defaults to ``"x3d_s"``.
        layer (str): Name of the layer to extract features from the backbone.
            Defaults to ``"blocks.4"``.
        pre_trained (bool, optional): Whether to use a pre-trained backbone.
            Defaults to ``True``.
        spatial_pool (bool, optional): Whether to use spatial pooling on features.
            Defaults to ``True``.
        pooling_kernel_size (int, optional): Kernel size for pooling features.
            Defaults to ``1``.
        pca_level (float, optional): Ratio of variance to preserve in PCA.
            Must be between 0 and 1.
            Defaults to ``0.98``.
        task (TaskType|str, optional): Whether to perform anomaly segmentation.
            If segmentation, perform segmentation along with detection
            Defaults to ``segmentation``.
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
        backbone: str = "x3d_s",
        layer: str = "blocks.4",
        pre_trained: bool = True,
        spatial_pool: bool = True,
        pooling_kernel_size: int = 1,
        pca_level: float = 0.98,
        task: TaskType|str = 'segmentation',
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

        self.model: FUVASModel = FUVASModel(
            backbone=backbone,
            pre_trained=pre_trained,
            layer=layer,
            pooling_kernel_size=pooling_kernel_size,
            n_comps=pca_level,
            task=task,
            spatial_pool=spatial_pool,
        )
        self.embeddings: list[torch.Tensor] = []

    @staticmethod
    def configure_optimizers() -> None:  # pylint: disable=arguments-differ
        """Configure optimizers for training.

        Returns:
            None: FUVAS doesn't require optimization.
        """
        return

    def training_step(self, batch: Batch, *args, **kwargs) -> torch.Tensor:
        """Extract features from the input batch during training.

        Args:
            batch (Batch): Input batch containing video clips.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            torch.Tensor: Dummy loss tensor for compatibility.
        """
        del args, kwargs  # These variables are not used.

        # Ensure batch.image is a tensor
        if batch.image is None or not isinstance(batch.image, torch.Tensor):
            msg = "Expected batch.image to be a tensor, but got None or non-tensor type"
            raise ValueError(msg)

        embedding = self.model.get_features(batch.image)[0].squeeze()
        self.embeddings.append(embedding)

        # Return a dummy loss tensor
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def fit(self) -> None:
        """Fit the PCA transformation to the embeddings.

        The method aggregates embeddings collected during training and fits
        the PCA transformation used for anomaly scoring.
        """
        logger.info("Aggregating the embedding extracted from the training set.")
        embeddings = torch.vstack(self.embeddings)

        logger.info("Fitting a PCA to dataset.")
        self.model.fit(embeddings)

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Compute predictions for the input batch during validation.

        Args:
            batch (Batch): Input batch containing video clips.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            STEP_OUTPUT: Dictionary containing anomaly scores and maps.
        """
        del args, kwargs  # These variables are not used.

        predictions = self.model(batch.image)
        return batch.update(pred_score=predictions.pred_score, anomaly_map=predictions.anomaly_map)

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Get FUVAS-specific trainer arguments.

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
