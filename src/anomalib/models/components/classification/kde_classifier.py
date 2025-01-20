"""Kernel Density Estimation Classifier.

This module provides a classifier based on kernel density estimation (KDE) for
anomaly detection. The classifier fits a KDE model to feature embeddings and uses
it to compute anomaly probabilities.

Example:
    >>> from anomalib.models.components.classification import KDEClassifier
    >>> from anomalib.models.components.classification import FeatureScalingMethod
    >>> # Create classifier with default settings
    >>> classifier = KDEClassifier()
    >>> # Create classifier with custom settings
    >>> classifier = KDEClassifier(
    ...     n_pca_components=32,
    ...     feature_scaling_method=FeatureScalingMethod.NORM,
    ...     max_training_points=50000
    ... )
    >>> # Fit classifier on embeddings
    >>> embeddings = torch.randn(1000, 512)  # Example embeddings
    >>> classifier.fit(embeddings)
    >>> # Get anomaly probabilities for new samples
    >>> new_embeddings = torch.randn(10, 512)
    >>> probabilities = classifier.predict(new_embeddings)
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import random
from enum import Enum

import torch
from torch import nn

from anomalib.models.components import PCA, GaussianKDE

logger = logging.getLogger(__name__)


class FeatureScalingMethod(str, Enum):
    """Feature scaling methods for KDE classifier.

    The scaling method determines how feature embeddings are normalized before
    being passed to the KDE model.

    Attributes:
        NORM: Scale features to unit vector length (L2 normalization)
        SCALE: Scale features by maximum length observed during training
            (preserves relative magnitudes)
    """

    NORM = "norm"  # scale to unit vector length
    SCALE = "scale"  # scale to max length observed in training


class KDEClassifier(nn.Module):
    """Classification module for KDE-based anomaly detection.

    This classifier uses kernel density estimation to model the distribution of
    normal samples in feature space. It first applies dimensionality reduction
    via PCA, then fits a Gaussian KDE model to the reduced features.

    Args:
        n_pca_components: Number of PCA components to retain. Lower values reduce
            computational cost but may lose information.
            Defaults to 16.
        feature_scaling_method: Method used to scale features before KDE.
            Options are ``norm`` (unit vector) or ``scale`` (max length).
            Defaults to ``FeatureScalingMethod.SCALE``.
        max_training_points: Maximum number of points used to fit the KDE model.
            If more points are provided, a random subset is selected.
            Defaults to 40000.

    Attributes:
        pca_model: PCA model for dimensionality reduction
        kde_model: Gaussian KDE model for density estimation
        max_length: Maximum feature length observed during training
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
        """Pre-process feature embeddings before KDE.

        Scales the features according to the specified scaling method.

        Args:
            feature_stack: Features extracted from the model, shape (N, D)
            max_length: Maximum feature length for scaling. If ``None``, computed
                from ``feature_stack``. Defaults to None.

        Returns:
            tuple: (scaled_features, max_length)

        Raises:
            RuntimeError: If unknown scaling method is specified
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
        """Fit the KDE classifier to training embeddings.

        Applies PCA, scales the features, and fits the KDE model.

        Args:
            embeddings: Training embeddings of shape (N, D)

        Returns:
            bool: True if fitting succeeded, False if insufficient samples

        Example:
            >>> classifier = KDEClassifier()
            >>> embeddings = torch.randn(1000, 512)
            >>> success = classifier.fit(embeddings)
            >>> assert success
        """
        if embeddings.shape[0] < self.n_pca_components:
            logger.info("Not enough features to commit. Not making a model.")
            return False

        # if max training points is non-zero and smaller than number of staged features, select random subset
        if embeddings.shape[0] > self.max_training_points:
            selected_idx = torch.tensor(
                random.sample(range(embeddings.shape[0]), self.max_training_points),
                device=embeddings.device,
            )
            selected_features = embeddings[selected_idx]
        else:
            selected_features = embeddings

        feature_stack = self.pca_model.fit_transform(selected_features)
        feature_stack, max_length = self.pre_process(feature_stack)
        self.max_length = max_length
        self.kde_model.fit(feature_stack)

        return True

    def compute_kde_scores(self, features: torch.Tensor, as_log_likelihood: bool | None = False) -> torch.Tensor:
        """Compute KDE scores for input features.

        Transforms features via PCA and scaling, then computes KDE scores.

        Args:
            features: Input features of shape (N, D)
            as_log_likelihood: If True, returns log of KDE scores.
                Defaults to False.

        Returns:
            torch.Tensor: KDE scores of shape (N,)
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
        """Convert density scores to anomaly probabilities.

        Uses sigmoid function to map scores to [0,1] range.
        See https://www.desmos.com/calculator/ifju7eesg7

        Args:
            scores: Density scores of shape (N,)

        Returns:
            torch.Tensor: Anomaly probabilities of shape (N,)
        """
        return 1 / (1 + torch.exp(0.05 * (scores - 12)))

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Predict anomaly probabilities for input features.

        Computes KDE scores and converts them to probabilities.

        Args:
            features: Input features of shape (N, D)

        Returns:
            torch.Tensor: Anomaly probabilities of shape (N,)

        Example:
            >>> classifier = KDEClassifier()
            >>> features = torch.randn(10, 512)
            >>> classifier.fit(features)
            >>> probs = classifier.predict(features)
            >>> assert probs.shape == (10,)
            >>> assert (probs >= 0).all() and (probs <= 1).all()
        """
        scores = self.compute_kde_scores(features, as_log_likelihood=True)
        return self.compute_probabilities(scores)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass of the classifier.

        Equivalent to calling ``predict()``.

        Args:
            features: Input features of shape (N, D)

        Returns:
            torch.Tensor: Anomaly probabilities of shape (N,)
        """
        return self.predict(features)
