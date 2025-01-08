"""Random Sparse Projector.

This module provides a PyTorch implementation of Sparse Random Projection for
dimensionality reduction.

Example:
    >>> import torch
    >>> from anomalib.models.components import SparseRandomProjection
    >>> # Create sample data
    >>> data = torch.randn(100, 50)  # 100 samples, 50 features
    >>> # Initialize projector
    >>> projector = SparseRandomProjection(eps=0.1)
    >>> # Fit and transform the data
    >>> projected_data = projector.fit_transform(data)
    >>> print(projected_data.shape)
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from sklearn.utils.random import sample_without_replacement


class NotFittedError(ValueError, AttributeError):
    """Exception raised when model is used before fitting."""


class SparseRandomProjection:
    """Sparse Random Projection using PyTorch operations.

    This class implements sparse random projection for dimensionality reduction
    using PyTorch. The implementation is based on the paper by Li et al. [1]_.

    Args:
        eps (float, optional): Minimum distortion rate parameter for calculating
            Johnson-Lindenstrauss minimum dimensions. Defaults to ``0.1``.
        random_state (int | None, optional): Seed for random number generation.
            Used for reproducible results. Defaults to ``None``.

    Attributes:
        n_components (int): Number of components in the projected space.
        sparse_random_matrix (torch.Tensor): Random projection matrix.
        eps (float): Minimum distortion rate.
        random_state (int | None): Random seed.

    Example:
        >>> import torch
        >>> from anomalib.models.components import SparseRandomProjection
        >>> # Create sample data
        >>> data = torch.randn(100, 50)  # 100 samples, 50 features
        >>> # Initialize and fit projector
        >>> projector = SparseRandomProjection(eps=0.1)
        >>> projector.fit(data)
        >>> # Transform data
        >>> projected = projector.transform(data)
        >>> print(projected.shape)

    References:
        .. [1] P. Li, T. Hastie and K. Church, "Very Sparse Random Projections,"
           KDD '06, 2006.
           https://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf
    """

    def __init__(self, eps: float = 0.1, random_state: int | None = None) -> None:
        self.n_components: int
        self.sparse_random_matrix: torch.Tensor
        self.eps = eps
        self.random_state = random_state

    def _sparse_random_matrix(self, n_features: int) -> torch.Tensor:
        """Generate a sparse random matrix for projection.

        Implements the sparse random matrix generation described in [1]_.

        Args:
            n_features (int): Dimensionality of the original source space.

        Returns:
            torch.Tensor: Sparse matrix of shape ``(n_components, n_features)``.
                The matrix is stored in dense format for GPU compatibility.

        References:
            .. [1] P. Li, T. Hastie and K. Church, "Very Sparse Random
               Projections," KDD '06, 2006.
        """
        # Density 'auto'. Factorize density
        density = 1 / np.sqrt(n_features)

        if density == 1:
            # skip index generation if totally dense
            binomial = torch.distributions.Binomial(total_count=1, probs=0.5)
            components = binomial.sample((self.n_components, n_features)) * 2 - 1
            components = 1 / np.sqrt(self.n_components) * components

        else:
            # Sparse matrix is not being generated here as it is stored as dense anyways
            components = torch.zeros((self.n_components, n_features), dtype=torch.float32)
            for i in range(self.n_components):
                # find the indices of the non-zero components for row i
                nnz_idx = torch.distributions.Binomial(total_count=n_features, probs=density).sample()
                # get nnz_idx column indices
                # pylint: disable=not-callable
                c_idx = torch.tensor(
                    sample_without_replacement(
                        n_population=n_features,
                        n_samples=nnz_idx,
                        random_state=self.random_state,
                    ),
                    dtype=torch.int32,
                )
                data = torch.distributions.Binomial(total_count=1, probs=0.5).sample(sample_shape=c_idx.size()) * 2 - 1
                # assign data to only those columns
                components[i, c_idx] = data

            components *= np.sqrt(1 / density) / np.sqrt(self.n_components)

        return components

    @staticmethod
    def _johnson_lindenstrauss_min_dim(n_samples: int, eps: float = 0.1) -> int | np.integer:
        """Find a 'safe' number of components for random projection.

        Implements the Johnson-Lindenstrauss lemma to determine the minimum number
        of components needed to approximately preserve distances.

        Args:
            n_samples (int): Number of samples in the dataset.
            eps (float, optional): Minimum distortion rate. Defaults to ``0.1``.

        Returns:
            int: Minimum number of components required.

        References:
            .. [1] Dasgupta, S. and Gupta, A., "An elementary proof of a theorem
               of Johnson and Lindenstrauss," Random Struct. Algor., 22: 60-65,
               2003.
        """
        denominator = (eps**2 / 2) - (eps**3 / 3)
        return (4 * np.log(n_samples) / denominator).astype(np.int64)

    def fit(self, embedding: torch.Tensor) -> "SparseRandomProjection":
        """Fit the random projection matrix to the data.

        Args:
            embedding (torch.Tensor): Input tensor of shape
                ``(n_samples, n_features)``.

        Returns:
            SparseRandomProjection: The fitted projector.

        Example:
            >>> projector = SparseRandomProjection()
            >>> data = torch.randn(100, 50)
            >>> projector = projector.fit(data)
        """
        n_samples, n_features = embedding.shape
        device = embedding.device

        self.n_components = self._johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=self.eps)

        # Generate projection matrix
        # torch can't multiply directly on sparse matrix and moving sparse matrix to cuda throws error
        # (Could not run 'aten::empty_strided' with arguments from the 'SparseCsrCUDA' backend)
        # hence sparse matrix is stored as a dense matrix on the device
        self.sparse_random_matrix = self._sparse_random_matrix(n_features=n_features).to(device)

        return self

    def transform(self, embedding: torch.Tensor) -> torch.Tensor:
        """Project the data using the random projection matrix.

        Args:
            embedding (torch.Tensor): Input tensor of shape
                ``(n_samples, n_features)``.

        Returns:
            torch.Tensor: Projected tensor of shape
                ``(n_samples, n_components)``.

        Raises:
            NotFittedError: If transform is called before fitting.

        Example:
            >>> projector = SparseRandomProjection()
            >>> data = torch.randn(100, 50)
            >>> projector.fit(data)
            >>> projected = projector.transform(data)
            >>> print(projected.shape)
        """
        if self.sparse_random_matrix is None:
            msg = "`fit()` has not been called on SparseRandomProjection yet."
            raise NotFittedError(msg)

        return embedding @ self.sparse_random_matrix.T.float()
