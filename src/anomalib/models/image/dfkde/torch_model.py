"""PyTorch model for Deep Feature Kernel Density Estimation (DFKDE).

This module provides a PyTorch implementation of the DFKDE model for anomaly
detection. The model extracts deep features from images using a pre-trained CNN
backbone and fits a kernel density estimation on these features to model the
distribution of normal samples.

Example:
    >>> import torch
    >>> from anomalib.models.image.dfkde.torch_model import DfkdeModel
    >>> model = DfkdeModel(
    ...     backbone="resnet18",
    ...     layers=["layer4"],
    ...     pre_trained=True
    ... )
    >>> batch = torch.randn(32, 3, 224, 224)
    >>> features = model(batch)  # Returns features during training
    >>> predictions = model(batch)  # Returns scores during inference

Notes:
    The model uses a pre-trained backbone to extract features and fits a KDE
    classifier on the embeddings during training. No gradient updates are
    performed on the backbone.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.data import InferenceBatch
from anomalib.models.components import TimmFeatureExtractor
from anomalib.models.components.classification import FeatureScalingMethod, KDEClassifier

logger = logging.getLogger(__name__)


class DfkdeModel(nn.Module):
    """Deep Feature Kernel Density Estimation model for anomaly detection.

    The model extracts deep features from images using a pre-trained CNN backbone
    and fits a kernel density estimation on these features to model the
    distribution of normal samples.

    Args:
        backbone (str): Name of the pre-trained model backbone from timm.
        layers (Sequence[str]): Names of layers to extract features from.
        pre_trained (bool, optional): Whether to use pre-trained backbone weights.
            Defaults to ``True``.
        n_pca_components (int, optional): Number of components for PCA dimension
            reduction. Defaults to ``16``.
        feature_scaling_method (FeatureScalingMethod, optional): Method used to
            scale features before KDE. Defaults to
            ``FeatureScalingMethod.SCALE``.
        max_training_points (int, optional): Maximum number of points used to fit
            the KDE model. Defaults to ``40000``.

    Example:
        >>> import torch
        >>> model = DfkdeModel(
        ...     backbone="resnet18",
        ...     layers=["layer4"],
        ...     pre_trained=True
        ... )
        >>> batch = torch.randn(32, 3, 224, 224)
        >>> features = model(batch)
    """

    def __init__(
        self,
        backbone: str,
        layers: Sequence[str],
        pre_trained: bool = True,
        n_pca_components: int = 16,
        feature_scaling_method: FeatureScalingMethod = FeatureScalingMethod.SCALE,
        max_training_points: int = 40000,
    ) -> None:
        super().__init__()

        self.feature_extractor = TimmFeatureExtractor(backbone=backbone, pre_trained=pre_trained, layers=layers).eval()

        self.classifier = KDEClassifier(
            n_pca_components=n_pca_components,
            feature_scaling_method=feature_scaling_method,
            max_training_points=max_training_points,
        )

    def get_features(self, batch: torch.Tensor) -> torch.Tensor:
        """Extract features from the pre-trained backbone network.

        Args:
            batch (torch.Tensor): Batch of input images with shape
                ``(N, C, H, W)``.

        Returns:
            torch.Tensor: Concatenated features from specified layers, flattened
                to shape ``(N, D)`` where ``D`` is the total feature dimension.

        Example:
            >>> batch = torch.randn(32, 3, 224, 224)
            >>> features = model.get_features(batch)
            >>> features.shape
            torch.Size([32, 512])  # Depends on backbone and layers
        """
        self.feature_extractor.eval()
        layer_outputs = self.feature_extractor(batch)
        for layer in layer_outputs:
            batch_size = len(layer_outputs[layer])
            layer_outputs[layer] = F.adaptive_avg_pool2d(input=layer_outputs[layer], output_size=(1, 1))
            layer_outputs[layer] = layer_outputs[layer].view(batch_size, -1)
        return torch.cat(list(layer_outputs.values())).detach()

    def forward(self, batch: torch.Tensor) -> torch.Tensor | InferenceBatch:
        """Extract features during training or compute anomaly scores during inference.

        Args:
            batch (torch.Tensor): Batch of input images with shape
                ``(N, C, H, W)``.

        Returns:
            torch.Tensor | InferenceBatch: During training, returns extracted
                features as a tensor. During inference, returns an
                ``InferenceBatch`` containing anomaly scores.

        Example:
            >>> batch = torch.randn(32, 3, 224, 224)
            >>> # Training mode
            >>> model.train()
            >>> features = model(batch)
            >>> # Inference mode
            >>> model.eval()
            >>> predictions = model(batch)
            >>> predictions.pred_score.shape
            torch.Size([32])
        """
        # 1. apply feature extraction
        features = self.get_features(batch)
        if self.training:
            return features

        # 2. apply density estimation
        scores = self.classifier(features)
        return InferenceBatch(pred_score=scores)
