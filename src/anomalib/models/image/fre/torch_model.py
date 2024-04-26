"""PyTorch model for DFM model implementation."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import math

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.models.components import PCA, DynamicBufferMixin, TimmFeatureExtractor


class TiedAE(nn.Module):

    def __init__(self, fullSz, projSz) -> None:
        super().__init__()
        self.fullSz = fullSz
        self.projSz = projSz
        self.weight = nn.Parameter(torch.empty(projSz, fullSz))
        torch.nn.init.xavier_uniform_(self.weight)
        self.decoder_bias = nn.Parameter(torch.zeros(fullSz))
        self.encoder_bias = nn.Parameter(torch.zeros(projSz))

    def encoder(self, input):
        encoded = F.linear(input, self.weight, self.encoder_bias)
        return encoded

    def decoder(self, input):
        decoded = F.linear(input, self.weight.t(), self.decoder_bias)
        return decoded

    def forward(self, input):
        encoded = F.linear(input, self.weight, self.encoder_bias)
        decoded = F.linear(encoded, self.weight.t(), self.decoder_bias)
        return decoded



class FREModel(nn.Module):
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

    # def fit(self, dataset: torch.Tensor) -> None:
    #     """Fit a pca transformation and a Gaussian model to dataset.

    #     Args:
    #         dataset (torch.Tensor): Input dataset to fit the model.
    #     """
    #     self.pca_model.fit(dataset)
    #     if self.score_type == "nll":
    #         features_reduced = self.pca_model.transform(dataset)
    #         self.gaussian_model.fit(features_reduced.T)

    # def score(self, features: torch.Tensor, feature_shapes: tuple) -> torch.Tensor:
    #     """Compute scores.

    #     Scores are either PCA-based feature reconstruction error (FRE) scores or
    #     the Gaussian density-based NLL scores

    #     Args:
    #         features (torch.Tensor): semantic features on which PCA and density modeling is performed.
    #         feature_shapes  (tuple): shape of `features` tensor. Used to generate anomaly map of correct shape.

    #     Returns:
    #         score (torch.Tensor): numpy array of scores
    #     """
    #     feats_projected = self.pca_model.transform(features)
    #     if self.score_type == "nll":
    #         score = self.gaussian_model.score_samples(feats_projected)
    #     elif self.score_type == "fre":
    #         feats_reconstructed = self.pca_model.inverse_transform(feats_projected)
    #         fre = torch.square(features - feats_reconstructed).reshape(feature_shapes)
    #         score_map = torch.unsqueeze(torch.sum(fre, dim=1), 1)
    #         score = torch.sum(torch.square(features - feats_reconstructed), dim=1)
    #     else:
    #         msg = f"unsupported score type: {self.score_type}"
    #         raise ValueError(msg)

    #     return (score, None) if self.score_type == "nll" else (score, score_map)

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
        # return (features_in, features_out)  if self.training else (features_in, feature_shapes)
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
