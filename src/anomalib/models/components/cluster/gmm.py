import logging

import torch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.functional import one_hot

from anomalib.models.components.base import DynamicBufferModule
from anomalib.models.components.cluster.kmeans import KMeans

logger = logging.getLogger(__name__)


class GaussianMixture(DynamicBufferModule):
    def __init__(self, n_components, n_iter=100, tol=1e-3):
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

        self.init_means = None

    def fit(self, X):
        # Initialize parameters with K-means
        self._initialize_parameters_kmeans(X)

        log_likelihood_old = 0
        converged = False
        for _ in range(self.n_iter):
            # E Step
            resp = self._e_step(X)
            # M Step
            self._m_step(X, resp)

            # Check for convergence
            log_likelihood_new = self._compute_log_likelihood(X).sum()
            if torch.abs(log_likelihood_new - log_likelihood_old) < self.tol:
                converged = True
                break
            log_likelihood_old = log_likelihood_new

        if not converged:
            logger.warn(
                f"GMM did not converge after {self.n_iter} iterations.  \
                        Consider increasing the number of iterations."
            )

    def _initialize_parameters_kmeans(self, X):
        """Initialize parameters with K-means."""
        labels, _ = KMeans(n_clusters=self.n_components).fit(X)
        resp = one_hot(labels, num_classes=self.n_components).float()
        self._m_step(X, resp)
        self.init_means = self.means

    def _e_step(self, X):
        log_resp = torch.stack(
            [
                MultivariateNormal(self.means[comp], self.covariances[comp]).log_prob(X)
                for comp in range(self.n_components)
            ],
            dim=1,
        )
        log_resp += torch.log(self.weights)
        log_resp -= torch.logsumexp(log_resp, dim=1, keepdim=True)
        return torch.exp(log_resp)

    def _m_step(self, X, resp):
        cluster_counts = resp.sum(axis=0)  # number of points in each cluster
        self.weights = resp.mean(axis=0)  # new weights
        self.means = (resp.T @ X) / cluster_counts[:, None]  # new means

        diff = X.unsqueeze(0) - self.means.unsqueeze(1)
        weighted_diff = diff * resp.T.unsqueeze(-1)
        covariances = torch.bmm(weighted_diff.transpose(-2, -1), diff) / cluster_counts.view(-1, 1, 1)
        # Add a small constant for numerical stability
        self.covariances = covariances + torch.eye(X.shape[1], device=X.device) * 1e-6  # new covariances

    def _compute_log_likelihood(self, X):
        log_likelihood = [
            MultivariateNormal(self.means[comp], self.covariances[comp]).log_prob(X)
            for comp in range(self.n_components)
        ]
        log_likelihood = torch.logsumexp(self.weights * torch.vstack(log_likelihood).T, dim=1)
        return log_likelihood

    def score_samples(self, X):
        return self._compute_log_likelihood(X)

    def predict(self, X):
        resp = self._e_step(X)
        return torch.argmax(resp, axis=1)


import torch
from sklearn import datasets
from sklearn.mixture import GaussianMixture as GMMSK

iris_dataset = datasets.load_iris()

# Example usage
torch.manual_seed(42)
X = torch.tensor(iris_dataset.data, dtype=torch.float32)

gmm = GaussianMixture(n_components=3, n_iter=100, tol=1e-3)
gmm.fit(X)
torch_labels = gmm.predict(X)

gmm_sk = GMMSK(n_components=3, max_iter=100, tol=1e-3, covariance_type="full", means_init=gmm.init_means)
gmm_sk.fit(X)
sk_labels = gmm_sk.predict(X)

# # assert all(torch_labels.numpy() == sk_labels)
