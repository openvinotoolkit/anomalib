"""PyTorch implementation of Gaussian Mixture Model.

This module provides a PyTorch-based implementation of Gaussian Mixture Model (GMM)
for clustering data into multiple Gaussian distributions.
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.functional import one_hot

from anomalib.models.components.base import DynamicBufferMixin
from anomalib.models.components.cluster.kmeans import KMeans

logger = logging.getLogger(__name__)


class GaussianMixture(DynamicBufferMixin):
    """Gaussian Mixture Model for clustering data into Gaussian distributions.

    Args:
        n_components (int): Number of Gaussian components to fit.
        n_iter (int, optional): Maximum number of EM iterations. Defaults to 100.
        tol (float, optional): Convergence threshold for log-likelihood.
            Defaults to 1e-3.

    Attributes:
        means (torch.Tensor): Means of the Gaussian components.
            Shape: ``(n_components, n_features)``.
        covariances (torch.Tensor): Covariance matrices of components.
            Shape: ``(n_components, n_features, n_features)``.
        weights (torch.Tensor): Mixing weights of components.
            Shape: ``(n_components,)``.

    Example:
        >>> import torch
        >>> from anomalib.models.components.cluster import GaussianMixture
        >>> # Create synthetic data with two clusters
        >>> data = torch.tensor([
        ...     [2, 1], [2, 2], [2, 3],  # Cluster 1
        ...     [7, 5], [8, 5], [9, 5],  # Cluster 2
        ... ]).float()
        >>> # Initialize and fit GMM
        >>> model = GaussianMixture(n_components=2)
        >>> model.fit(data)
        >>> # Get cluster means
        >>> model.means
        tensor([[8., 5.],
                [2., 2.]])
        >>> # Predict cluster assignments
        >>> model.predict(data)
        tensor([1, 1, 1, 0, 0, 0])
        >>> # Get log-likelihood scores
        >>> model.score_samples(data)
        tensor([3.8295, 4.5795, 3.8295, 3.8295, 4.5795, 3.8295])
    """

    def __init__(self, n_components: int, n_iter: int = 100, tol: float = 1e-3) -> None:
        super().__init__()
        self.n_components = n_components
        self.tol = tol
        self.n_iter = n_iter

        self.register_buffer("means", torch.Tensor())
        self.register_buffer("covariances", torch.Tensor())
        self.register_buffer("weights", torch.Tensor())

        self.means: torch.Tensor
        self.covariances: torch.Tensor
        self.weights: torch.Tensor

    def fit(self, data: torch.Tensor) -> None:
        """Fit the GMM to the input data using EM algorithm.

        Args:
            data (torch.Tensor): Input data to fit the model to.
                Shape: ``(n_samples, n_features)``.
        """
        self._initialize_parameters_kmeans(data)

        log_likelihood_old = 0
        converged = False
        for _ in range(self.n_iter):
            # E-step
            log_likelihood_new, resp = self._e_step(data)
            # M-step
            self._m_step(data, resp)

            # Check for convergence
            if torch.abs(log_likelihood_new - log_likelihood_old) < self.tol:
                converged = True
                break
            log_likelihood_old = log_likelihood_new

        if not converged:
            logger.warning(
                f"GMM did not converge after {self.n_iter} iterations. Consider increasing the number of iterations.",
            )

    def _initialize_parameters_kmeans(self, data: torch.Tensor) -> None:
        """Initialize GMM parameters using K-means clustering.

        Args:
            data (torch.Tensor): Input data for initialization.
                Shape: ``(n_samples, n_features)``.
        """
        labels, _ = KMeans(n_clusters=self.n_components).fit(data)
        resp = one_hot(labels, num_classes=self.n_components).float()
        self._m_step(data, resp)

    def _e_step(self, data: torch.Tensor) -> torch.Tensor:
        """Perform E-step to compute responsibilities and log-likelihood.

        Args:
            data (torch.Tensor): Input data.
                Shape: ``(n_samples, n_features)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - Mean log-likelihood of the data
                - Responsibilities for each component.
                  Shape: ``(n_samples, n_components)``
        """
        weighted_log_prob = self._estimate_weighted_log_prob(data)
        log_prob_norm = torch.logsumexp(weighted_log_prob, axis=1)
        log_resp = weighted_log_prob - torch.logsumexp(
            weighted_log_prob,
            dim=1,
            keepdim=True,
        )
        return torch.mean(log_prob_norm), torch.exp(log_resp)

    def _m_step(self, data: torch.Tensor, resp: torch.Tensor) -> None:
        """Perform M-step to update GMM parameters.

        Args:
            data (torch.Tensor): Input data.
                Shape: ``(n_samples, n_features)``.
            resp (torch.Tensor): Responsibilities from E-step.
                Shape: ``(n_samples, n_components)``.
        """
        cluster_counts = resp.sum(axis=0)  # number of points in each cluster
        self.weights = resp.mean(axis=0)  # new weights
        self.means = (resp.T @ data) / cluster_counts[:, None]  # new means

        diff = data.unsqueeze(0) - self.means.unsqueeze(1)
        weighted_diff = diff * resp.T.unsqueeze(-1)
        covariances = torch.bmm(
            weighted_diff.transpose(-2, -1),
            diff,
        ) / cluster_counts.view(-1, 1, 1)
        # Add a small constant for numerical stability
        self.covariances = (
            covariances
            + torch.eye(
                data.shape[1],
                device=data.device,
            )
            * 1e-6
        )

    def _estimate_weighted_log_prob(self, data: torch.Tensor) -> torch.Tensor:
        """Estimate weighted log probabilities for each component.

        Args:
            data (torch.Tensor): Input data.
                Shape: ``(n_samples, n_features)``.

        Returns:
            torch.Tensor: Weighted log probabilities.
                Shape: ``(n_samples, n_components)``.
        """
        log_prob = torch.stack(
            [
                MultivariateNormal(
                    self.means[comp],
                    self.covariances[comp],
                ).log_prob(data)
                for comp in range(self.n_components)
            ],
            dim=1,
        )
        return log_prob + torch.log(self.weights)

    def score_samples(self, data: torch.Tensor) -> torch.Tensor:
        """Compute per-sample likelihood scores.

        Args:
            data (torch.Tensor): Input samples to score.
                Shape: ``(n_samples, n_features)``.

        Returns:
            torch.Tensor: Log-likelihood scores.
                Shape: ``(n_samples,)``.
        """
        return torch.logsumexp(self._estimate_weighted_log_prob(data), dim=1)

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """Predict cluster assignments for the input data.

        Args:
            data (torch.Tensor): Input samples.
                Shape: ``(n_samples, n_features)``.

        Returns:
            torch.Tensor: Predicted cluster labels.
                Shape: ``(n_samples,)``.
        """
        _, resp = self._e_step(data)
        return torch.argmax(resp, axis=1)
