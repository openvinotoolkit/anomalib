"""Gaussian Kernel Density Estimation.

This module implements non-parametric density estimation using Gaussian kernels.
The bandwidth is selected automatically using Scott's rule.

Example:
    >>> import torch
    >>> from anomalib.models.components.stats import GaussianKDE
    >>> # Create density estimator
    >>> kde = GaussianKDE()
    >>> # Fit and evaluate density
    >>> features = torch.randn(100, 10)  # 100 samples, 10 dimensions
    >>> kde.fit(features)
    >>> density = kde.predict(features)
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math

import torch

from anomalib.models.components.base import DynamicBufferMixin


class GaussianKDE(DynamicBufferMixin):
    """Gaussian Kernel Density Estimation.

    Estimates probability density using a Gaussian kernel function. The bandwidth
    is selected automatically using Scott's rule.

    Args:
        dataset (torch.Tensor | None, optional): Dataset on which to fit the KDE
            model. If provided, the model will be fitted immediately.
            Defaults to ``None``.

    Example:
        >>> import torch
        >>> from anomalib.models.components.stats import GaussianKDE
        >>> features = torch.randn(100, 10)  # 100 samples, 10 dimensions
        >>> # Initialize and fit in one step
        >>> kde = GaussianKDE(dataset=features)
        >>> # Or fit later
        >>> kde = GaussianKDE()
        >>> kde.fit(features)
        >>> # Get density estimates
        >>> density = kde(features)
    """

    def __init__(self, dataset: torch.Tensor | None = None) -> None:
        super().__init__()

        if dataset is not None:
            self.fit(dataset)

        self.register_buffer("bw_transform", torch.Tensor())
        self.register_buffer("dataset", torch.Tensor())
        self.register_buffer("norm", torch.Tensor())

        self.bw_transform = torch.Tensor()
        self.dataset = torch.Tensor()
        self.norm = torch.Tensor()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute KDE estimates for the input features.

        Args:
            features (torch.Tensor): Feature tensor of shape ``(N, D)`` where
                ``N`` is the number of samples and ``D`` is the dimension.

        Returns:
            torch.Tensor: Density estimates for each input sample, shape ``(N,)``.

        Example:
            >>> kde = GaussianKDE()
            >>> features = torch.randn(100, 10)
            >>> kde.fit(features)
            >>> estimates = kde(features)
            >>> estimates.shape
            torch.Size([100])
        """
        features = torch.matmul(features, self.bw_transform)

        estimate = torch.zeros(features.shape[0]).to(features.device)
        for i in range(features.shape[0]):
            embedding = ((self.dataset - features[i]) ** 2).sum(dim=1)
            embedding = torch.exp(-embedding / 2) * self.norm
            estimate[i] = torch.mean(embedding)

        return estimate

    def fit(self, dataset: torch.Tensor) -> None:
        """Fit the KDE model to the input dataset.

        Computes the bandwidth matrix using Scott's rule and transforms the data
        accordingly.

        Args:
            dataset (torch.Tensor): Input dataset of shape ``(N, D)`` where ``N``
                is the number of samples and ``D`` is the dimension.

        Example:
            >>> kde = GaussianKDE()
            >>> features = torch.randn(100, 10)
            >>> kde.fit(features)
        """
        num_samples, dimension = dataset.shape

        # compute scott's bandwidth factor
        factor = num_samples ** (-1 / (dimension + 4))

        cov_mat = self.cov(dataset.T)
        inv_cov_mat = torch.linalg.inv(cov_mat)
        inv_cov = inv_cov_mat / factor**2

        # transform data to account for bandwidth
        bw_transform = torch.linalg.cholesky(inv_cov)
        dataset = torch.matmul(dataset, bw_transform)

        norm = torch.prod(torch.diag(bw_transform))
        norm *= math.pow((2 * math.pi), (-dimension / 2))

        self.bw_transform = bw_transform
        self.dataset = dataset
        self.norm = norm

    @staticmethod
    def cov(tensor: torch.Tensor) -> torch.Tensor:
        """Calculate the unbiased covariance matrix.

        Args:
            tensor (torch.Tensor): Input tensor of shape ``(D, N)`` where ``D``
                is the dimension and ``N`` is the number of samples.

        Returns:
            torch.Tensor: Covariance matrix of shape ``(D, D)``.

        Example:
            >>> x = torch.randn(5, 100)  # 5 dimensions, 100 samples
            >>> cov_matrix = GaussianKDE.cov(x)
            >>> cov_matrix.shape
            torch.Size([5, 5])
        """
        mean = torch.mean(tensor, dim=1)
        tensor -= mean[:, None]
        return torch.matmul(tensor, tensor.T) / (tensor.size(1) - 1)
