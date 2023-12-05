"""Pytorch implementation of Gaussian Mixture Model."""
import logging

import torch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.functional import one_hot

from anomalib.models.components.base import DynamicBufferModule
from anomalib.models.components.cluster.kmeans import KMeans

logger = logging.getLogger(__name__)


class GaussianMixture(DynamicBufferModule):
    """Gaussian Mixture Model.

    Args:
        n_components (int): Number of components.
        n_iter (int): Maximum number of iterations to perform.
        tol (float): Convergence threshold.
    """

    def __init__(self, n_components: int, n_iter: int = 100, tol: float = 1e-3) -> None:
        super().__init__()
        self.n_components = n_components
        self.tol = tol
        self.n_iter = n_iter

        self.register_buffer("means", Tensor())
        self.register_buffer("covariances", Tensor())
        self.register_buffer("weights", Tensor())

        self.means: Tensor
        self.covariances: Tensor
        self.weights: Tensor
        self.precisions_cholesky: Tensor

    def fit(self, data: Tensor) -> None:
        """Fit the model to the data.

        Args:
            data (Tensor): Data to fit the model to. Tensor of shape (n_samples, n_features).
        """
        self._initialize_parameters_kmeans(data)

        log_likelihood_old = 0
        converged = False
        for _ in range(self.n_iter):
            # E-step
            resp = self._e_step(data)
            # M-step
            self._m_step(data, resp)

            # Check for convergence
            log_likelihood_new = self._compute_log_likelihood(data).sum()
            if torch.abs(log_likelihood_new - log_likelihood_old) < self.tol:
                converged = True
                break
            log_likelihood_old = log_likelihood_new

        if not converged:
            logger.warning(
                f"GMM did not converge after {self.n_iter} iterations.  \
                        Consider increasing the number of iterations.",
            )

    def _initialize_parameters_kmeans(self, data: Tensor) -> None:
        """Initialize parameters with K-means.

        Args:
            data (Tensor): Data to fit the model to. Tensor of shape (n_samples, n_features).
        """
        labels, _ = KMeans(n_clusters=self.n_components).fit(data)
        resp = one_hot(labels, num_classes=self.n_components).float()
        self._m_step(data, resp)

    def _e_step(self, data: Tensor) -> Tensor:
        """Perform the E-step to estimate the responsibilities of the gaussians.

        Args:
            data (Tensor): Data to fit the model to. Tensor of shape (n_samples, n_features).

        Returns:
            Tensor: Tensor of shape (n_samples, n_components) containing the responsibilities.
        """
        log_resp = torch.stack(
            [
                MultivariateNormal(self.means[comp], self.covariances[comp]).log_prob(data)
                for comp in range(self.n_components)
            ],
            dim=1,
        )
        log_resp += torch.log(self.weights)
        log_resp -= torch.logsumexp(log_resp, dim=1, keepdim=True)
        return torch.exp(log_resp)

    def _m_step(self, data: Tensor, resp: Tensor) -> None:
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

    def _compute_log_likelihood(self, data: Tensor) -> Tensor:
        """Compute the log-likelihood of the data given the gaussian parameters.

        Args:
            data (Tensor): Data to fit the model to. Tensor of shape (n_samples, n_features).

        Returns:
            Tensor: Tensor of shape (n_samples,) containing the log-likelihood of each sample.
        """
        log_likelihood = [
            MultivariateNormal(self.means[comp], self.covariances[comp]).log_prob(data)
            for comp in range(self.n_components)
        ]
        return torch.logsumexp(self.weights * torch.vstack(log_likelihood).T, dim=1)

    def score_samples(self, data: Tensor) -> Tensor:
        """Assign a likelihood score to each sample in the data.

        Args:
            data (Tensor): Samples to assign scores to. Tensor of shape (n_samples, n_features).

        Returns:
            Tensor: Tensor of shape (n_samples,) containing the log-likelihood score of each sample.
        """
        return self._compute_log_likelihood(data)

    def predict(self, data: Tensor) -> Tensor:
        """Predict the cluster labels of the data.

        Args:
            data (Tensor): Samples to assign to clusters. Tensor of shape (n_samples, n_features).

        Returns:
            Tensor: Tensor of shape (n_samples,) containing the predicted cluster label of each sample.
        """
        resp = self._e_step(data)
        return torch.argmax(resp, axis=1)
