"""Multi Variate Gaussian Distribution.

This module implements parametric density estimation using a multivariate Gaussian
distribution. It estimates the mean and covariance matrix from input features.

Example:
    >>> import torch
    >>> from anomalib.models.components.stats import MultiVariateGaussian
    >>> # Create distribution estimator
    >>> mvg = MultiVariateGaussian()
    >>> # Fit distribution to features
    >>> features = torch.randn(100, 64, 32, 32)  # B x C x H x W
    >>> mean, inv_cov = mvg.fit(features)
    >>> # Access distribution parameters
    >>> print(mean.shape)      # [64, 1024]
    >>> print(inv_cov.shape)   # [1024, 64, 64]
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from torch import nn

from anomalib.models.components.base import DynamicBufferMixin


class MultiVariateGaussian(DynamicBufferMixin, nn.Module):
    """Multi Variate Gaussian Distribution.

    Estimates a multivariate Gaussian distribution by computing the mean and
    covariance matrix from input feature embeddings. The distribution parameters
    are stored as buffers.

    Example:
        >>> import torch
        >>> from anomalib.models.components.stats import MultiVariateGaussian
        >>> mvg = MultiVariateGaussian()
        >>> features = torch.randn(100, 64, 32, 32)  # B x C x H x W
        >>> mean, inv_cov = mvg.fit(features)
        >>> print(mean.shape)      # [64, 1024]
        >>> print(inv_cov.shape)   # [1024, 64, 64]
    """

    def __init__(self) -> None:
        """Initialize empty buffers for mean and inverse covariance."""
        super().__init__()

        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("inv_covariance", torch.empty(0))

        self.mean: torch.Tensor
        self.inv_covariance: torch.Tensor

    @staticmethod
    def _cov(
        observations: torch.Tensor,
        rowvar: bool = False,
        bias: bool = False,
        ddof: int | None = None,
        aweights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Estimate covariance matrix similar to numpy.cov.

        Args:
            observations: A 1-D or 2-D tensor containing multiple variables and
                observations. Each row represents a variable, and each column a
                single observation of all variables if ``rowvar=True``. The
                relationship is transposed if ``rowvar=False``.
            rowvar: If ``True``, each row represents a variable. If ``False``,
                each column represents a variable. Defaults to ``False``.
            bias: If ``False`` (default), normalize by ``(N-1)`` for unbiased
                estimate. If ``True``, normalize by ``N``. Can be overridden by
                ``ddof``.
            ddof: Delta degrees of freedom. If not ``None``, overrides ``bias``.
                ``ddof=1`` gives unbiased estimate, ``ddof=0`` gives simple
                average.
            aweights: Optional 1-D tensor of observation weights. Larger weights
                indicate more "important" observations. If ``ddof=0``, weights
                are treated as observation probabilities.

        Returns:
            Covariance matrix of the variables.
        """
        # ensure at least 2D
        if observations.dim() == 1:
            observations = observations.view(-1, 1)

        # treat each column as a data point, each row as a variable
        if rowvar and observations.shape[0] != 1:
            observations = observations.t()

        if ddof is None:
            ddof = 1 if bias == 0 else 0

        weights = aweights
        weights_sum: Any

        if weights is not None:
            if not torch.is_tensor(weights):
                weights = torch.tensor(weights, dtype=torch.float)
            weights_sum = torch.sum(weights)
            avg = torch.sum(observations * (weights / weights_sum)[:, None], 0)
        else:
            avg = torch.mean(observations, 0)

        # Determine the normalization
        if weights is None:
            fact = observations.shape[0] - ddof
        elif ddof == 0:
            fact = weights_sum
        elif aweights is None:
            fact = weights_sum - ddof
        else:
            fact = weights_sum - ddof * torch.sum(weights * weights) / weights_sum

        observations_m = observations.sub(avg.expand_as(observations))

        x_transposed = observations_m.t() if weights is None else torch.mm(torch.diag(weights), observations_m).t()

        covariance = torch.mm(x_transposed, observations_m)
        covariance = covariance / fact

        return covariance.squeeze()

    def forward(self, embedding: torch.Tensor) -> list[torch.Tensor]:
        """Calculate multivariate Gaussian distribution parameters.

        Computes the mean and inverse covariance matrix from input feature
        embeddings. A small regularization term (0.01) is added to the diagonal
        of the covariance matrix for numerical stability.

        Args:
            embedding: Input tensor of shape ``(B, C, H, W)`` containing CNN
                feature embeddings.

        Returns:
            List containing:
                - Mean tensor of shape ``(C, H*W)``
                - Inverse covariance tensor of shape ``(H*W, C, C)``
        """
        device = embedding.device

        batch, channel, height, width = embedding.size()
        embedding_vectors = embedding.view(batch, channel, height * width)
        self.mean = torch.mean(embedding_vectors, dim=0)
        covariance = torch.zeros(size=(channel, channel, height * width), device=device)
        identity = torch.eye(channel).to(device)
        for i in range(height * width):
            covariance[:, :, i] = self._cov(embedding_vectors[:, :, i], rowvar=False) + 0.01 * identity

        # Stabilize the covariance matrix by adding a small regularization term
        stabilized_covariance = covariance.permute(2, 0, 1) + 1e-5 * identity

        # Check if the device is MPS and fallback to CPU if necessary
        if device.type == "mps":
            # Move stabilized covariance to CPU for inversion
            self.inv_covariance = torch.linalg.inv(stabilized_covariance.cpu()).to(device)
        else:
            # Calculate inverse covariance as we need only the inverse
            self.inv_covariance = torch.linalg.inv(stabilized_covariance)

        return [self.mean, self.inv_covariance]

    def fit(self, embedding: torch.Tensor) -> list[torch.Tensor]:
        """Fit multivariate Gaussian distribution to input embeddings.

        Convenience method that calls ``forward()`` to compute distribution
        parameters.

        Args:
            embedding: Input tensor of shape ``(B, C, H, W)`` containing CNN
                feature embeddings.

        Returns:
            List containing the mean and inverse covariance tensors.
        """
        return self.forward(embedding)
