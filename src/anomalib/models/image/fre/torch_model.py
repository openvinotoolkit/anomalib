"""PyTorch model for DFM model implementation."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import math

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.models.components import PCA, DynamicBufferMixin, TimmFeatureExtractor


class TiedAE(nn.Module):
    """Model for the Tied AutoEncoder used for FRE calculation.

    Args:
        full_size (int): Dimension of input to the tied auto-encoder.
        proj_size (int): Dimension of the reduced-dimension latent space of the tied auto-encoder.
    """

    def __init__(self, full_size, proj_size) -> None:
        super().__init__()
        self.full_size = full_size
        self.proj_size = proj_size
        self.weight = nn.Parameter(torch.empty(proj_size, full_size))
        torch.nn.init.xavier_uniform_(self.weight)
        self.decoder_bias = nn.Parameter(torch.zeros(full_size))
        self.encoder_bias = nn.Parameter(torch.zeros(proj_size))

    def forward(self, features_in) -> torch.Tensor:
        """Run input features through the autoencoder.

        Args:
            features_in (torch.Tensor): Feature batch.

        Returns:
            Tensor: torch.Tensor containing reconstructed features.
        """
        encoded = F.linear(features_in, self.weight, self.encoder_bias)
        return F.linear(encoded, self.weight.t(), self.decoder_bias)


class FREModel(nn.Module):
    """Model for the FRE algorithm.

    Args:
        backbone (str): Pre-trained model backbone.
        layer (str): Layer from which to extract features.
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
            Defaults to ``True``.
        pooling_kernel_size (int, optional): Kernel size to pool features extracted from the CNN.
            Defaults to ``4``.
        full_size (int, optional): Dimension of feature at output of layer specified in layer.
            Defaults to ``65536``. 
        proj_size (int, optional): Reduced size of feature after applying dimensionality reduction
            via shallow linear autoencoder.
            Defaults to ``220``. 
    """

    def __init__(
        self,
        backbone: str,
        layer: str,
        full_size: int,
        proj_size: int,
        pre_trained: bool = True,
        pooling_kernel_size: int = 4
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pooling_kernel_size = pooling_kernel_size
        self.fre_model = TiedAE(full_size, proj_size)
        self.layer = layer
        self.feature_extractor = TimmFeatureExtractor(
            backbone=self.backbone,
            pre_trained=pre_trained,
            layers=[layer],
        ).eval()

    def get_features(self, batch: torch.Tensor) -> torch.Tensor:
        """Extract features from the pretrained network.

        Args:
            batch (torch.Tensor): Image batch.

        Returns:
            Tensor: torch.Tensor containing extracted features.
        """
        self.feature_extractor.eval()
        features_in = self.feature_extractor(batch)[self.layer]
        batch_size = len(features_in)
        if self.pooling_kernel_size > 1:
            features_in = F.avg_pool2d(input=features_in, kernel_size=self.pooling_kernel_size)
        feature_shapes = features_in.shape
        features_in = features_in.view(batch_size, -1).detach()
        features_out = self.fre_model(features_in)
        return features_in, features_out, feature_shapes

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Compute score from input images.

        Args:
            batch (torch.Tensor): Input images

        Returns:
            Tensor: Scores
        """
        features_in, features_out, feature_shapes = self.get_features(batch)
        fre = torch.square(features_in - features_out).reshape(feature_shapes)
        anomaly_map = torch.sum(fre, 1)  # NxCxHxW --> NxHxW                
        score = torch.sum(anomaly_map, (1,2))  # NxHxW --> N
        anomaly_map = torch.unsqueeze(anomaly_map, 1)
        anomaly_map = F.interpolate(anomaly_map, size=batch.shape[-2:], mode="bilinear", align_corners=False)
        return score, anomaly_map
