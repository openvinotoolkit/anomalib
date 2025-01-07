"""PyTorch model for Deep Feature Modeling (DFM).

This module provides a PyTorch implementation of the DFM model for anomaly
detection. The model extracts deep features from images using a pre-trained CNN
backbone and fits a Gaussian model on these features to detect anomalies.

Example:
    >>> import torch
    >>> from anomalib.models.image.dfm.torch_model import DFMModel
    >>> model = DFMModel(
    ...     backbone="resnet18",
    ...     layer="layer4",
    ...     pre_trained=True
    ... )
    >>> batch = torch.randn(32, 3, 224, 224)
    >>> features = model(batch)  # Returns features during training
    >>> predictions = model(batch)  # Returns scores during inference

Notes:
    The model uses a pre-trained backbone to extract features and fits a PCA
    transformation followed by a Gaussian model during training. No gradient
    updates are performed on the backbone.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.data import InferenceBatch
from anomalib.models.components import PCA, DynamicBufferMixin, TimmFeatureExtractor


class SingleClassGaussian(DynamicBufferMixin):
    """Model Gaussian distribution over a set of points.

    This class fits a single Gaussian distribution to a set of feature vectors
    and computes likelihood scores for new samples.

    Example:
        >>> gaussian = SingleClassGaussian()
        >>> features = torch.randn(128, 100)  # 100 samples of 128 dimensions
        >>> gaussian.fit(features)
        >>> scores = gaussian.score_samples(features)
    """

    def __init__(self) -> None:
        """Initialize Gaussian model with empty buffers."""
        super().__init__()
        self.register_buffer("mean_vec", torch.Tensor())
        self.register_buffer("u_mat", torch.Tensor())
        self.register_buffer("sigma_mat", torch.Tensor())

        self.mean_vec: torch.Tensor
        self.u_mat: torch.Tensor
        self.sigma_mat: torch.Tensor

    def fit(self, dataset: torch.Tensor) -> None:
        """Fit a Gaussian model to dataset X.

        Covariance matrix is not calculated directly using ``C = X.X^T``.
        Instead, it is represented using SVD of X: ``X = U.S.V^T``.
        Hence, ``C = U.S^2.U^T``. This simplifies the calculation of the
        log-likelihood without requiring full matrix inversion.

        Args:
            dataset (torch.Tensor): Input dataset to fit the model with shape
                ``(n_features, n_samples)``.
        """
        num_samples = dataset.shape[1]
        self.mean_vec = torch.mean(dataset, dim=1, device=dataset.device)
        data_centered = (dataset - self.mean_vec.reshape(-1, 1)) / math.sqrt(num_samples)
        self.u_mat, self.sigma_mat, _ = torch.linalg.svd(data_centered, full_matrices=False)

    def score_samples(self, features: torch.Tensor) -> torch.Tensor:
        """Compute the negative log likelihood (NLL) scores.

        Args:
            features (torch.Tensor): Semantic features on which density modeling
                is performed with shape ``(n_samples, n_features)``.

        Returns:
            torch.Tensor: NLL scores for each sample.
        """
        features_transformed = torch.matmul(features - self.mean_vec, self.u_mat / self.sigma_mat)
        return torch.sum(features_transformed * features_transformed, dim=1) + 2 * torch.sum(torch.log(self.sigma_mat))

    def forward(self, dataset: torch.Tensor) -> None:
        """Fit the model to the input dataset.

        Transforms the input dataset based on singular values calculated earlier.

        Args:
            dataset (torch.Tensor): Input dataset with shape
                ``(n_features, n_samples)``.
        """
        self.fit(dataset)


class DFMModel(nn.Module):
    """Deep Feature Modeling (DFM) model for anomaly detection.

    The model extracts deep features from images using a pre-trained CNN backbone
    and fits a Gaussian model on these features to detect anomalies.

    Args:
        backbone (str): Pre-trained model backbone from timm.
        layer (str): Layer from which to extract features.
        pre_trained (bool, optional): Whether to use pre-trained backbone.
            Defaults to ``True``.
        pooling_kernel_size (int, optional): Kernel size to pool features.
            Defaults to ``4``.
        n_comps (float, optional): Ratio for PCA components calculation.
            Defaults to ``0.97``.
        score_type (str, optional): Scoring type - ``fre`` or ``nll``.
            Defaults to ``fre``. Segmentation supported with ``fre`` only.
            For ``nll``, set task to classification.

    Example:
        >>> model = DFMModel(
        ...     backbone="resnet18",
        ...     layer="layer4",
        ...     pre_trained=True
        ... )
        >>> input_tensor = torch.randn(32, 3, 224, 224)
        >>> output = model(input_tensor)
    """

    def __init__(
        self,
        backbone: str,
        layer: str,
        pre_trained: bool = True,
        pooling_kernel_size: int = 4,
        n_comps: float = 0.97,
        score_type: str = "fre",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pooling_kernel_size = pooling_kernel_size
        self.n_components = n_comps
        self.pca_model = PCA(n_components=self.n_components)
        self.gaussian_model = SingleClassGaussian()
        self.score_type = score_type
        self.layer = layer
        self.feature_extractor = TimmFeatureExtractor(
            backbone=self.backbone,
            pre_trained=pre_trained,
            layers=[layer],
        ).eval()

    def fit(self, dataset: torch.Tensor) -> None:
        """Fit PCA and Gaussian model to dataset.

        Args:
            dataset (torch.Tensor): Input dataset with shape
                ``(n_samples, n_features)``.
        """
        self.pca_model.fit(dataset)
        if self.score_type == "nll":
            features_reduced = self.pca_model.transform(dataset)
            self.gaussian_model.fit(features_reduced.T)

    def score(self, features: torch.Tensor, feature_shapes: tuple) -> torch.Tensor:
        """Compute anomaly scores.

        Scores are either PCA-based feature reconstruction error (FRE) scores or
        Gaussian density-based NLL scores.

        Args:
            features (torch.Tensor): Features for scoring with shape
                ``(n_samples, n_features)``.
            feature_shapes (tuple): Shape of features tensor for anomaly map.

        Returns:
            tuple[torch.Tensor, Optional[torch.Tensor]]: Tuple containing
                (scores, anomaly_maps). Anomaly maps are None for NLL scoring.
        """
        feats_projected = self.pca_model.transform(features)
        if self.score_type == "nll":
            score = self.gaussian_model.score_samples(feats_projected)
        elif self.score_type == "fre":
            feats_reconstructed = self.pca_model.inverse_transform(feats_projected)
            fre = torch.square(features - feats_reconstructed).reshape(feature_shapes)
            score_map = torch.unsqueeze(torch.sum(fre, dim=1), 1)
            score = torch.sum(torch.square(features - feats_reconstructed), dim=1)
        else:
            msg = f"unsupported score type: {self.score_type}"
            raise ValueError(msg)

        return (score, None) if self.score_type == "nll" else (score, score_map)

    def get_features(self, batch: torch.Tensor) -> torch.Tensor:
        """Extract features from the pretrained network.

        Args:
            batch (torch.Tensor): Input images with shape
                ``(batch_size, channels, height, width)``.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Size]]: Features during
                training, or tuple of (features, feature_shapes) during inference.
        """
        self.feature_extractor.eval()
        features = self.feature_extractor(batch)[self.layer]
        batch_size = len(features)
        if self.pooling_kernel_size > 1:
            features = F.avg_pool2d(input=features, kernel_size=self.pooling_kernel_size)
        feature_shapes = features.shape
        features = features.view(batch_size, -1).detach()
        return features if self.training else (features, feature_shapes)

    def forward(self, batch: torch.Tensor) -> torch.Tensor | InferenceBatch:
        """Compute anomaly predictions from input images.

        Args:
            batch (torch.Tensor): Input images with shape
                ``(batch_size, channels, height, width)``.

        Returns:
            Union[torch.Tensor, InferenceBatch]: Model predictions. During
                training returns features tensor. During inference returns
                ``InferenceBatch`` with prediction scores and anomaly maps.
        """
        feature_vector, feature_shapes = self.get_features(batch)
        pred_score, anomaly_map = self.score(feature_vector.view(feature_vector.shape[:2]), feature_shapes)
        if anomaly_map is not None:
            anomaly_map = F.interpolate(anomaly_map, size=batch.shape[-2:], mode="bilinear", align_corners=False)
        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
