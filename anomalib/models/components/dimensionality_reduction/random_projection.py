"""This module comprises PatchCore Sampling Methods for the embedding.

- Random Sparse Projector
    Sparse Random Projection using PyTorch Operations
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import numpy as np
import torch
from sklearn.utils.random import sample_without_replacement
from torch import Tensor


class NotFittedError(ValueError, AttributeError):
    """Raise Exception if estimator is used before fitting."""


class SparseRandomProjection:
    """Sparse Random Projection using PyTorch operations.

    Args:
        eps (float, optional): Minimum distortion rate parameter for calculating
            Johnson-Lindenstrauss minimum dimensions. Defaults to 0.1.
        random_state (Optional[int], optional): Uses the seed to set the random
            state for sample_without_replacement function. Defaults to None.
    """

    def __init__(self, eps: float = 0.1, random_state: Optional[int] = None) -> None:
        self.n_components: int
        self.sparse_random_matrix: Tensor
        self.eps = eps
        self.random_state = random_state

    def _sparse_random_matrix(self, n_features: int):
        """Random sparse matrix. Based on https://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf.

        Args:
            n_features (int): Dimentionality of the original source space

        Returns:
            Tensor: Sparse matrix of shape (n_components, n_features).
                The generated Gaussian random matrix is in CSR (compressed sparse row)
                format.
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
            components = torch.zeros((self.n_components, n_features), dtype=torch.float64)
            for i in range(self.n_components):
                # find the indices of the non-zero components for row i
                nnz_idx = torch.distributions.Binomial(total_count=n_features, probs=density).sample()
                # get nnz_idx column indices
                # pylint: disable=not-callable
                c_idx = torch.tensor(
                    sample_without_replacement(
                        n_population=n_features, n_samples=nnz_idx, random_state=self.random_state
                    ),
                    dtype=torch.int64,
                )
                data = torch.distributions.Binomial(total_count=1, probs=0.5).sample(sample_shape=c_idx.size()) * 2 - 1
                # assign data to only those columns
                components[i, c_idx] = data.double()

            components *= np.sqrt(1 / density) / np.sqrt(self.n_components)

        return components

    def johnson_lindenstrauss_min_dim(self, n_samples: int, eps: float = 0.1):
        """Find a 'safe' number of components to randomly project to.

        Ref eqn 2.1 https://cseweb.ucsd.edu/~dasgupta/papers/jl.pdf

        Args:
            n_samples (int): Number of samples used to compute safe components
            eps (float, optional): Minimum distortion rate. Defaults to 0.1.
        """

        denominator = (eps**2 / 2) - (eps**3 / 3)
        return (4 * np.log(n_samples) / denominator).astype(np.int64)

    def fit(self, embedding: Tensor) -> "SparseRandomProjection":
        """Generates sparse matrix from the embedding tensor.

        Args:
            embedding (Tensor): embedding tensor for generating embedding

        Returns:
            (SparseRandomProjection): Return self to be used as
            >>> generator = SparseRandomProjection()
            >>> generator = generator.fit()
        """
        n_samples, n_features = embedding.shape
        device = embedding.device

        self.n_components = self.johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=self.eps)

        # Generate projection matrix
        # torch can't multiply directly on sparse matrix and moving sparse matrix to cuda throws error
        # (Could not run 'aten::empty_strided' with arguments from the 'SparseCsrCUDA' backend)
        # hence sparse matrix is stored as a dense matrix on the device
        self.sparse_random_matrix = self._sparse_random_matrix(n_features=n_features).to(device)

        return self

    def transform(self, embedding: Tensor) -> Tensor:
        """Project the data by using matrix product with the random matrix.

        Args:
            embedding (Tensor): Embedding of shape (n_samples, n_features)
                The input data to project into a smaller dimensional space

        Returns:
            projected_embedding (Tensor): Sparse matrix of shape
                (n_samples, n_components) Projected array.
        """
        if self.sparse_random_matrix is None:
            raise NotFittedError("`fit()` has not been called on SparseRandomProjection yet.")

        projected_embedding = embedding @ self.sparse_random_matrix.T.float()
        return projected_embedding
