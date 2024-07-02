"""PyTorch model for DFM model implementation."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.models.components import PCA, DynamicBufferMixin, TimmFeatureExtractor


class SingleClassGaussian(DynamicBufferMixin):
    """Model Gaussian distribution over a set of points."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean_vec", torch.Tensor())
        self.register_buffer("u_mat", torch.Tensor())
        self.register_buffer("sigma_mat", torch.Tensor())

        self.mean_vec: torch.Tensor
        self.u_mat: torch.Tensor
        self.sigma_mat: torch.Tensor

    def fit(self, dataset: torch.Tensor) -> None:
        """Fit a Gaussian model to dataset X.

        Covariance matrix is not calculated directly using:
        ``C = X.X^T``
        Instead, it is represented in terms of the Singular Value Decomposition of X:
        ``X = U.S.V^T``
        Hence,
        ``C = U.S^2.U^T``
        This simplifies the calculation of the log-likelihood without requiring full matrix inversion.

        Args:
            dataset (torch.Tensor): Input dataset to fit the model.
        """
        num_samples = dataset.shape[1]
        self.mean_vec = torch.mean(dataset, dim=1)
        data_centered = (dataset - self.mean_vec.reshape(-1, 1)) / math.sqrt(num_samples)
        self.u_mat, self.sigma_mat, _ = torch.linalg.svd(data_centered, full_matrices=False)

    def score_samples(self, features: torch.Tensor) -> torch.Tensor:
        """Compute the NLL (negative log likelihood) scores.

        Args:
            features (torch.Tensor): semantic features on which density modeling is performed.

        Returns:
            nll (torch.Tensor): Torch tensor of scores
        """
        features_transformed = torch.matmul(features - self.mean_vec, self.u_mat / self.sigma_mat)
        return torch.sum(features_transformed * features_transformed, dim=1) + 2 * torch.sum(torch.log(self.sigma_mat))

    def forward(self, dataset: torch.Tensor) -> None:
        """Provide the same functionality as `fit`.

        Transforms the input dataset based on singular values calculated earlier.

        Args:
            dataset (torch.Tensor): Input dataset
        """
        self.fit(dataset)


class DFMModel(nn.Module):
    """Model for the DFM algorithm.

    Args:
        backbone (str): Pre-trained model backbone.
        layer (str): Layer from which to extract features.
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
            Defaults to ``True``.
        pooling_kernel_size (int, optional): Kernel size to pool features extracted from the CNN.
            Defaults to ``4``.
        n_comps (float, optional): Ratio from which number of components for PCA are calculated.
            Defaults to ``0.97``.
        score_type (str, optional): Scoring type. Options are `fre` and `nll`.  Anomaly
            Defaults to ``fre``. Segmentation is supported with `fre` only.
            If using `nll`, set `task` in config.yaml to classification Defaults to ``classification``.
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
        """Fit a pca transformation and a Gaussian model to dataset.

        Args:
            dataset (torch.Tensor): Input dataset to fit the model.
        """
        self.pca_model.fit(dataset)
        if self.score_type == "nll":
            features_reduced = self.pca_model.transform(dataset)
            self.gaussian_model.fit(features_reduced.T)

    def score(self, features: torch.Tensor, feature_shapes: tuple) -> torch.Tensor:
        """Compute scores.

        Scores are either PCA-based feature reconstruction error (FRE) scores or
        the Gaussian density-based NLL scores

        Args:
            features (torch.Tensor): semantic features on which PCA and density modeling is performed.
            feature_shapes  (tuple): shape of `features` tensor. Used to generate anomaly map of correct shape.

        Returns:
            score (torch.Tensor): numpy array of scores
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
            batch (torch.Tensor): Image batch.

        Returns:
            Tensor: torch.Tensor containing extracted features.
        """
        self.feature_extractor.eval()
        features = self.feature_extractor(batch)[self.layer]
        batch_size = len(features)
        if self.pooling_kernel_size > 1:
            features = F.avg_pool2d(input=features, kernel_size=self.pooling_kernel_size)
        feature_shapes = features.shape
        features = features.view(batch_size, -1).detach()
        return features if self.training else (features, feature_shapes)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Compute score from input images.

        Args:
            batch (torch.Tensor): Input images

        Returns:
            Tensor: Scores
        """
        feature_vector, feature_shapes = self.get_features(batch)
        score, score_map = self.score(feature_vector.view(feature_vector.shape[:2]), feature_shapes)
        if score_map is not None:
            score_map = F.interpolate(score_map, size=batch.shape[-2:], mode="bilinear", align_corners=False)
        return score, score_map
