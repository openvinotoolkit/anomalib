"""Kernel Density Estimation Classifier."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import random
from enum import Enum

import torch
from torch import nn

from anomalib.models.components import PCA, GaussianKDE

logger = logging.getLogger(__name__)


class FeatureScalingMethod(str, Enum):
    """Determines how the feature embeddings are scaled."""

    NORM = "norm"  # scale to unit vector length
    SCALE = "scale"  # scale to max length observed in training (preserve relative magnitude)


class KDEClassifier(nn.Module):
    """Classification module for KDE-based anomaly detection.

    Args:
        n_pca_components (int, optional): Number of PCA components. Defaults to 16.
        feature_scaling_method (FeatureScalingMethod, optional): Scaling method applied to features before passing to
            KDE. Options are `norm` (normalize to unit vector length) and `scale` (scale to max length observed in
            training).
        max_training_points (int, optional): Maximum number of training points to fit the KDE model. Defaults to 40000.
    """

    def __init__(
        self,
        n_pca_components: int = 16,
        feature_scaling_method: FeatureScalingMethod = FeatureScalingMethod.SCALE,
        max_training_points: int = 40000,
    ) -> None:
        super().__init__()

        self.n_pca_components = n_pca_components
        self.feature_scaling_method = feature_scaling_method
        self.max_training_points = max_training_points

        self.pca_model = PCA(n_components=self.n_pca_components)
        self.kde_model = GaussianKDE()

        self.register_buffer("max_length", torch.empty([]))
        self.max_length = torch.empty([])

    def pre_process(
        self,
        feature_stack: torch.Tensor,
        max_length: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pre-process the CNN features.

        Args:
          feature_stack (torch.Tensor): Features extracted from CNN
          max_length (Tensor | None): Used to unit normalize the feature_stack vector. If ``max_len`` is not
            provided, the length is calculated from the ``feature_stack``. Defaults to None.

        Returns:
            (Tuple): Stacked features and length
        """
        if max_length is None:
            max_length = torch.max(torch.linalg.norm(feature_stack, ord=2, dim=1))

        if self.feature_scaling_method == FeatureScalingMethod.NORM:
            feature_stack /= torch.linalg.norm(feature_stack, ord=2, dim=1)[:, None]
        elif self.feature_scaling_method == FeatureScalingMethod.SCALE:
            feature_stack /= max_length
        else:
            msg = "Unknown pre-processing mode. Available modes are: Normalized and Scale."
            raise RuntimeError(msg)
        return feature_stack, max_length

    def fit(self, embeddings: torch.Tensor) -> bool:
        """Fit a kde model to embeddings.

        Args:
            embeddings (torch.Tensor): Input embeddings to fit the model.

        Returns:
            Boolean confirming whether the training is successful.
        """
        if embeddings.shape[0] < self.n_pca_components:
            logger.info("Not enough features to commit. Not making a model.")
            return False

        # if max training points is non-zero and smaller than number of staged features, select random subset
        if embeddings.shape[0] > self.max_training_points:
            selected_idx = torch.tensor(random.sample(range(embeddings.shape[0]), self.max_training_points))
            selected_features = embeddings[selected_idx]
        else:
            selected_features = embeddings

        feature_stack = self.pca_model.fit_transform(selected_features)
        feature_stack, max_length = self.pre_process(feature_stack)
        self.max_length = max_length
        self.kde_model.fit(feature_stack)

        return True

    def compute_kde_scores(self, features: torch.Tensor, as_log_likelihood: bool | None = False) -> torch.Tensor:
        """Compute the KDE scores.

        The scores calculated from the KDE model are converted to densities. If `as_log_likelihood` is set to true then
            the log of the scores are calculated.

        Args:
            features (torch.Tensor): Features to which the PCA model is fit.
            as_log_likelihood (bool | None, optional): If true, gets log likelihood scores. Defaults to False.

        Returns:
            (torch.Tensor): Score
        """
        features = self.pca_model.transform(features)
        features, _ = self.pre_process(features, self.max_length)
        # Scores are always assumed to be passed as a density
        kde_scores = self.kde_model(features)

        # add small constant to avoid zero division in log computation
        kde_scores += 1e-300

        if as_log_likelihood:
            kde_scores = torch.log(kde_scores)

        return kde_scores

    @staticmethod
    def compute_probabilities(scores: torch.Tensor) -> torch.Tensor:
        """Convert density scores to anomaly probabilities (see https://www.desmos.com/calculator/ifju7eesg7).

        Args:
          scores (torch.Tensor): density of an image.

        Returns:
          probability that image with {density} is anomalous
        """
        return 1 / (1 + torch.exp(0.05 * (scores - 12)))

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Predicts the probability that the features belong to the anomalous class.

        Args:
          features (torch.Tensor): Feature from which the output probabilities are detected.

        Returns:
          Detection probabilities
        """
        scores = self.compute_kde_scores(features, as_log_likelihood=True)
        return self.compute_probabilities(scores)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Make predictions on extracted features."""
        return self.predict(features)
