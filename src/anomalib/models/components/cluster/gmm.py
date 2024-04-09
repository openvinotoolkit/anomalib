"""Pytorch implementation of Gaussian Mixture Model."""

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
    """Gaussian Mixture Model.

    Args:
        n_components (int): Number of components.
        n_iter (int): Maximum number of iterations to perform.
            Defaults to ``100``.
        tol (float): Convergence threshold.
            Defaults to ``1e-3``.

    Example:
        The following examples shows how to fit a Gaussian Mixture Model to some data and get the cluster means and
        predicted labels and log-likelihood scores of the data.

        .. code-block:: python

            >>> import torch
            >>> from anomalib.models.components.cluster import GaussianMixture
            >>> model = GaussianMixture(n_components=2)
            >>> data = torch.tensor(
            ...     [
            ...             [2, 1], [2, 2], [2, 3],
            ...             [7, 5], [8, 5], [9, 5],
            ...     ]
            ... ).float()
            >>> model.fit(data)
            >>> model.means  # get the means of the gaussians
            tensor([[8., 5.],
                    [2., 2.]])
            >>> model.predict(data)  # get the predicted cluster label of each sample
            tensor([1, 1, 1, 0, 0, 0])
            >>> model.score_samples(data)  # get the log-likelihood score of each sample
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
        """Fit the model to the data.

        Args:
            data (Tensor): Data to fit the model to. Tensor of shape (n_samples, n_features).
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
                f"GMM did not converge after {self.n_iter} iterations.  \
                        Consider increasing the number of iterations.",
            )

    def _initialize_parameters_kmeans(self, data: torch.Tensor) -> None:
        """Initialize parameters with K-means.

        Args:
            data (Tensor): Data to fit the model to. Tensor of shape (n_samples, n_features).
        """
        labels, _ = KMeans(n_clusters=self.n_components).fit(data)
        resp = one_hot(labels, num_classes=self.n_components).float()
        self._m_step(data, resp)

    def _e_step(self, data: torch.Tensor) -> torch.Tensor:
        """Perform the E-step to estimate the responsibilities of the gaussians.

        Args:
            data (Tensor): Data to fit the model to. Tensor of shape (n_samples, n_features).

        Returns:
            Tensor: log probability of the data given the gaussians.
            Tensor: Tensor of shape (n_samples, n_components) containing the responsibilities.
        """
        weighted_log_prob = self._estimate_weighted_log_prob(data)
        log_prob_norm = torch.logsumexp(weighted_log_prob, axis=1)
        log_resp = weighted_log_prob - torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
        return torch.mean(log_prob_norm), torch.exp(log_resp)

    def _m_step(self, data: torch.Tensor, resp: torch.Tensor) -> None:
        """Perform the M-step to update the parameters of the gaussians.

        Args:
            data (Tensor): Data to fit the model to. Tensor of shape (n_samples, n_features).
            resp (Tensor): Tensor of shape (n_samples, n_components) containing the responsibilities.
        """
        cluster_counts = resp.sum(axis=0)  # number of points in each cluster
        self.weights = resp.mean(axis=0)  # new weights
        self.means = (resp.T @ data) / cluster_counts[:, None]  # new means

        diff = data.unsqueeze(0) - self.means.unsqueeze(1)
        weighted_diff = diff * resp.T.unsqueeze(-1)
        covariances = torch.bmm(weighted_diff.transpose(-2, -1), diff) / cluster_counts.view(-1, 1, 1)
        # Add a small constant for numerical stability
        self.covariances = covariances + torch.eye(data.shape[1], device=data.device) * 1e-6  # new covariances

    def _estimate_weighted_log_prob(self, data: torch.Tensor) -> torch.Tensor:
        """Estimate the log probability of the data given the gaussian parameters.

        Args:
            data (Tensor): Data to fit the model to. Tensor of shape (n_samples, n_features).

        Returns:
            Tensor: Tensor of shape (n_samples, n_components) containing the log-probabilities of each sample.
        """
        log_prob = torch.stack(
            [
                MultivariateNormal(self.means[comp], self.covariances[comp]).log_prob(data)
                for comp in range(self.n_components)
            ],
            dim=1,
        )
        return log_prob + torch.log(self.weights)

    def score_samples(self, data: torch.Tensor) -> torch.Tensor:
        """Assign a likelihood score to each sample in the data.

        Args:
            data (Tensor): Samples to assign scores to. Tensor of shape (n_samples, n_features).

        Returns:
            Tensor: Tensor of shape (n_samples,) containing the log-likelihood score of each sample.
        """
        return torch.logsumexp(self._estimate_weighted_log_prob(data), dim=1)

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """Predict the cluster labels of the data.

        Args:
            data (Tensor): Samples to assign to clusters. Tensor of shape (n_samples, n_features).

        Returns:
            Tensor: Tensor of shape (n_samples,) containing the predicted cluster label of each sample.
        """
        _, resp = self._e_step(data)
        return torch.argmax(resp, axis=1)
