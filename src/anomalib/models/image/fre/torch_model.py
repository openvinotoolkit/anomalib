"""PyTorch model for the Feature Reconstruction Error (FRE) algorithm implementation.

The FRE model extracts features from a pre-trained CNN backbone and learns to
reconstruct them using a tied autoencoder. Anomalies are detected by measuring
the reconstruction error between original and reconstructed features.

Example:
    >>> from anomalib.models.image.fre.torch_model import FREModel
    >>> model = FREModel(
    ...     backbone="resnet50",
    ...     layer="layer3",
    ...     input_dim=65536,
    ...     latent_dim=220,
    ...     pre_trained=True,
    ...     pooling_kernel_size=4
    ... )
    >>> input_tensor = torch.randn(32, 3, 256, 256)
    >>> output = model(input_tensor)
    >>> output.pred_score.shape
    torch.Size([32])
    >>> output.anomaly_map.shape
    torch.Size([32, 1, 256, 256])

Paper:
    Title: FRE: Feature Reconstruction Error for Unsupervised Anomaly Detection
           and Segmentation
    URL: https://papers.bmvc2023.org/0614.pdf

See Also:
    :class:`anomalib.models.image.fre.lightning_model.Fre`:
        PyTorch Lightning implementation of the FRE model.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.data import InferenceBatch
from anomalib.models.components import TimmFeatureExtractor


class TiedAE(nn.Module):
    """Tied Autoencoder used for feature reconstruction error calculation.

    The tied autoencoder uses shared weights between encoder and decoder to reduce
    the number of parameters while maintaining reconstruction capability.

    Args:
        input_dim (int): Dimension of input features to the tied autoencoder.
        latent_dim (int): Dimension of the reduced latent space representation.

    Example:
        >>> tied_ae = TiedAE(input_dim=1024, latent_dim=128)
        >>> features = torch.randn(32, 1024)
        >>> reconstructed = tied_ae(features)
        >>> reconstructed.shape
        torch.Size([32, 1024])
    """

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.weight = nn.Parameter(torch.empty(latent_dim, input_dim))
        torch.nn.init.xavier_uniform_(self.weight)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        self.encoder_bias = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run input features through the autoencoder.

        The features are first encoded to a lower dimensional latent space and
        then decoded back to the original feature space using transposed weights.

        Args:
            features (torch.Tensor): Input feature batch of shape
                ``(N, input_dim)``.

        Returns:
            torch.Tensor: Reconstructed features of shape ``(N, input_dim)``.
        """
        encoded = F.linear(features, self.weight, self.encoder_bias)
        return F.linear(encoded, self.weight.t(), self.decoder_bias)


class FREModel(nn.Module):
    """Feature Reconstruction Error (FRE) model implementation.

    The model extracts features from a pre-trained CNN backbone and learns to
    reconstruct them using a tied autoencoder. Anomalies are detected by
    measuring the reconstruction error between original and reconstructed
    features.

    Args:
        backbone (str): Pre-trained CNN backbone architecture (e.g.
            ``"resnet18"``, ``"resnet50"``, etc.).
        layer (str): Layer name from which to extract features (e.g.
            ``"layer2"``, ``"layer3"``, etc.).
        input_dim (int, optional): Dimension of features at output of specified
            layer.
            Defaults to ``65536``.
        latent_dim (int, optional): Reduced feature dimension after applying
            dimensionality reduction via shallow linear autoencoder.
            Defaults to ``220``.
        pre_trained (bool, optional): Whether to use pre-trained backbone
            weights.
            Defaults to ``True``.
        pooling_kernel_size (int, optional): Kernel size for pooling features
            extracted from the CNN.
            Defaults to ``4``.

    Example:
        >>> model = FREModel(
        ...     backbone="resnet50",
        ...     layer="layer3",
        ...     input_dim=65536,
        ...     latent_dim=220
        ... )
        >>> input_tensor = torch.randn(32, 3, 256, 256)
        >>> output = model(input_tensor)
        >>> output.pred_score.shape
        torch.Size([32])
        >>> output.anomaly_map.shape
        torch.Size([32, 1, 256, 256])
    """

    def __init__(
        self,
        backbone: str,
        layer: str,
        input_dim: int = 65536,
        latent_dim: int = 220,
        pre_trained: bool = True,
        pooling_kernel_size: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pooling_kernel_size = pooling_kernel_size
        self.fre_model = TiedAE(input_dim, latent_dim)
        self.layer = layer
        self.feature_extractor = TimmFeatureExtractor(
            backbone=self.backbone,
            pre_trained=pre_trained,
            layers=[layer],
        ).eval()

    def get_features(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract and reconstruct features from the pretrained network.

        Args:
            batch (torch.Tensor): Input image batch of shape
                ``(N, C, H, W)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing:
                - Original features of shape ``(N, D)``
                - Reconstructed features of shape ``(N, D)``
                - Original feature tensor shape ``(N, C, H, W)``
                where ``D`` is the flattened feature dimension.
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

    def forward(self, batch: torch.Tensor) -> InferenceBatch:
        """Generate anomaly predictions for input images.

        The method:
        1. Extracts and reconstructs features using the tied autoencoder
        2. Computes reconstruction error as anomaly scores
        3. Generates pixel-wise anomaly maps
        4. Upsamples anomaly maps to input image size

        Args:
            batch (torch.Tensor): Input image batch of shape
                ``(N, C, H, W)``.

        Returns:
            InferenceBatch: Batch containing:
                - Anomaly scores of shape ``(N,)``
                - Anomaly maps of shape ``(N, 1, H, W)``
        """
        features_in, features_out, feature_shapes = self.get_features(batch)
        fre = torch.square(features_in - features_out).reshape(feature_shapes)
        anomaly_map = torch.sum(fre, 1)  # NxCxHxW --> NxHxW
        score = torch.sum(anomaly_map, (1, 2))  # NxHxW --> N
        anomaly_map = torch.unsqueeze(anomaly_map, 1)
        anomaly_map = F.interpolate(anomaly_map, size=batch.shape[-2:], mode="bilinear", align_corners=False)
        return InferenceBatch(pred_score=score, anomaly_map=anomaly_map)
