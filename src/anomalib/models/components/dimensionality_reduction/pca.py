"""Principal Component Analysis (PCA) implementation using PyTorch.

This module provides a PyTorch-based implementation of Principal Component Analysis
for dimensionality reduction.

Example:
    >>> import torch
    >>> from anomalib.models.components import PCA
    >>> # Create sample data
    >>> data = torch.randn(100, 10)  # 100 samples, 10 features
    >>> # Initialize PCA with 3 components
    >>> pca = PCA(n_components=3)
    >>> # Fit and transform the data
    >>> transformed_data = pca.fit_transform(data)
    >>> print(transformed_data.shape)
    torch.Size([100, 3])
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from anomalib.models.components.base import DynamicBufferMixin


class PCA(DynamicBufferMixin):
    """Principal Component Analysis (PCA) for dimensionality reduction.

    Args:
        n_components (int | float): Number of components to keep. If float between
            0 and 1, represents the variance ratio to preserve. If int, represents
            the exact number of components to keep.

    Attributes:
        singular_vectors (torch.Tensor): Right singular vectors from SVD.
        singular_values (torch.Tensor): Singular values from SVD.
        mean (torch.Tensor): Mean of the training data.
        num_components (torch.Tensor): Number of components kept.

    Example:
        >>> import torch
        >>> from anomalib.models.components import PCA
        >>> # Create sample data
        >>> data = torch.randn(100, 10)  # 100 samples, 10 features
        >>> # Initialize with fixed number of components
        >>> pca = PCA(n_components=3)
        >>> pca.fit(data)
        >>> # Transform new data
        >>> transformed = pca.transform(data)
        >>> print(transformed.shape)
        torch.Size([100, 3])
        >>> # Initialize with variance ratio
        >>> pca = PCA(n_components=0.95)  # Keep 95% of variance
        >>> pca.fit(data)
    """

    def __init__(self, n_components: int | float) -> None:
        super().__init__()
        self.n_components = n_components

        self.register_buffer("singular_vectors", torch.Tensor())
        self.register_buffer("mean", torch.Tensor())
        self.register_buffer("num_components", torch.Tensor())

        self.singular_vectors: torch.Tensor
        self.singular_values: torch.Tensor
        self.mean: torch.Tensor
        self.num_components: torch.Tensor

    def fit(self, dataset: torch.Tensor) -> None:
        """Fit the PCA model to the dataset.

        Args:
            dataset (torch.Tensor): Input dataset of shape ``(n_samples,
                n_features)``.

        Example:
            >>> data = torch.randn(100, 10)
            >>> pca = PCA(n_components=3)
            >>> pca.fit(data)
            >>> # Access fitted attributes
            >>> print(pca.singular_vectors.shape)
            torch.Size([10, 3])
            >>> print(pca.mean.shape)
            torch.Size([10])
        """
        mean = dataset.mean(dim=0)
        dataset -= mean

        _, sig, v_h = torch.linalg.svd(dataset.double(), full_matrices=False)
        num_components: int
        if self.n_components <= 1:
            variance_ratios = torch.cumsum(sig * sig, dim=0) / torch.sum(sig * sig)
            num_components = torch.nonzero(variance_ratios >= self.n_components)[0]
        else:
            num_components = int(self.n_components)

        self.num_components = torch.tensor([num_components], device=dataset.device)

        self.singular_vectors = v_h.transpose(-2, -1)[:, :num_components].float()
        self.singular_values = sig[:num_components].float()
        self.mean = mean

    def fit_transform(self, dataset: torch.Tensor) -> torch.Tensor:
        """Fit the model and transform the input dataset.

        Args:
            dataset (torch.Tensor): Input dataset of shape ``(n_samples,
                n_features)``.

        Returns:
            torch.Tensor: Transformed dataset of shape ``(n_samples,
                n_components)``.

        Example:
            >>> data = torch.randn(100, 10)
            >>> pca = PCA(n_components=3)
            >>> transformed = pca.fit_transform(data)
            >>> print(transformed.shape)
            torch.Size([100, 3])
        """
        mean = dataset.mean(dim=0)
        dataset -= mean
        num_components = int(self.n_components)
        self.num_components = torch.tensor([num_components], device=dataset.device)

        v_h = torch.linalg.svd(dataset)[-1]
        self.singular_vectors = v_h.transpose(-2, -1)[:, :num_components]
        self.mean = mean

        return torch.matmul(dataset, self.singular_vectors)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """Transform features using the fitted PCA model.

        Args:
            features (torch.Tensor): Input features of shape ``(n_samples,
                n_features)``.

        Returns:
            torch.Tensor: Transformed features of shape ``(n_samples,
                n_components)``.

        Example:
            >>> data = torch.randn(100, 10)
            >>> pca = PCA(n_components=3)
            >>> pca.fit(data)
            >>> new_data = torch.randn(50, 10)
            >>> transformed = pca.transform(new_data)
            >>> print(transformed.shape)
            torch.Size([50, 3])
        """
        features -= self.mean
        return torch.matmul(features, self.singular_vectors)

    def inverse_transform(self, features: torch.Tensor) -> torch.Tensor:
        """Inverse transform features back to original space.

        Args:
            features (torch.Tensor): Transformed features of shape ``(n_samples,
                n_components)``.

        Returns:
            torch.Tensor: Reconstructed features of shape ``(n_samples,
                n_features)``.

        Example:
            >>> data = torch.randn(100, 10)
            >>> pca = PCA(n_components=3)
            >>> transformed = pca.fit_transform(data)
            >>> reconstructed = pca.inverse_transform(transformed)
            >>> print(reconstructed.shape)
            torch.Size([100, 10])
        """
        return torch.matmul(features, self.singular_vectors.transpose(-2, -1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Transform features (alias for transform method).

        Args:
            features (torch.Tensor): Input features of shape ``(n_samples,
                n_features)``.

        Returns:
            torch.Tensor: Transformed features of shape ``(n_samples,
                n_components)``.

        Example:
            >>> data = torch.randn(100, 10)
            >>> pca = PCA(n_components=3)
            >>> pca.fit(data)
            >>> transformed = pca(data)  # Using forward
            >>> print(transformed.shape)
            torch.Size([100, 3])
        """
        return self.transform(features)
