"""Normality model of DFKDE."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.models.components import TimmFeatureExtractor
from anomalib.models.components.classification import FeatureScalingMethod, KDEClassifier

logger = logging.getLogger(__name__)


class DfkdeModel(nn.Module):
    """Normality Model for the DFKDE algorithm.

    Args:
        backbone (str): Pre-trained model backbone.
        layers (Sequence[str]): Layers to extract features from.
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
            Defaults to ``True``.
        n_pca_components (int, optional): Number of PCA components.
            Defaults to ``16``.
        feature_scaling_method (FeatureScalingMethod, optional): Feature scaling method.
            Defaults to ``FeatureScalingMethod.SCALE``.
        max_training_points (int, optional): Number of training points to fit the KDE model.
            Defaults to ``40000``.
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
        """Extract features from the pretrained network.

        Args:
            batch (torch.Tensor): Image batch.

        Returns:
            Tensor: torch.Tensor containing extracted features.
        """
        self.feature_extractor.eval()
        layer_outputs = self.feature_extractor(batch)
        for layer in layer_outputs:
            batch_size = len(layer_outputs[layer])
            layer_outputs[layer] = F.adaptive_avg_pool2d(input=layer_outputs[layer], output_size=(1, 1))
            layer_outputs[layer] = layer_outputs[layer].view(batch_size, -1)
        return torch.cat(list(layer_outputs.values())).detach()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Prediction by normality model.

        Args:
            batch (torch.Tensor): Input images.

        Returns:
            Tensor: Predictions
        """
        # 1. apply feature extraction
        features = self.get_features(batch)
        if self.training:
            return features

        # 2. apply density estimation
        return self.classifier(features)
