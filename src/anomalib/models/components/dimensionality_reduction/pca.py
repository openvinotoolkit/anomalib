"""Principle Component Analysis (PCA) with PyTorch."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from anomalib.models.components.base import DynamicBufferMixin


class PCA(DynamicBufferMixin):
    """Principle Component Analysis (PCA).

    Args:
        n_components (float): Number of components. Can be either integer number of components
          or a ratio between 0-1.

    Example:
        >>> import torch
        >>> from anomalib.models.components import PCA

        Create a PCA model with 2 components:

        >>> pca = PCA(n_components=2)

        Create a random embedding and fit a PCA model.

        >>> embedding = torch.rand(1000, 5).cuda()
        >>> pca = PCA(n_components=2)
        >>> pca.fit(embedding)

        Apply transformation:

        >>> transformed = pca.transform(embedding)
        >>> transformed.shape
        torch.Size([1000, 2])
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
        """Fits the PCA model to the dataset.

        Args:
          dataset (torch.Tensor): Input dataset to fit the model.

        Example:
            >>> pca.fit(embedding)
            >>> pca.singular_vectors
            tensor([9.6053, 9.2763], device='cuda:0')

            >>> pca.mean
            tensor([0.4859, 0.4959, 0.4906, 0.5010, 0.5042], device='cuda:0')
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

        self.num_components = torch.Tensor([num_components])

        self.singular_vectors = v_h.transpose(-2, -1)[:, :num_components].float()
        self.singular_values = sig[:num_components].float()
        self.mean = mean

    def fit_transform(self, dataset: torch.Tensor) -> torch.Tensor:
        """Fit and transform PCA to dataset.

        Args:
            dataset (torch.Tensor): Dataset to which the PCA if fit and transformed

        Returns:
            Transformed dataset

        Example:
            >>> pca.fit_transform(embedding)
            >>> transformed_embedding = pca.fit_transform(embedding)
            >>> transformed_embedding.shape
            torch.Size([1000, 2])
        """
        mean = dataset.mean(dim=0)
        dataset -= mean
        num_components = int(self.n_components)
        self.num_components = torch.Tensor([num_components])

        v_h = torch.linalg.svd(dataset)[-1]
        self.singular_vectors = v_h.transpose(-2, -1)[:, :num_components]
        self.mean = mean

        return torch.matmul(dataset, self.singular_vectors)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """Transform the features based on singular vectors calculated earlier.

        Args:
            features (torch.Tensor): Input features

        Returns:
            Transformed features

        Example:
            >>> pca.transform(embedding)
            >>> transformed_embedding = pca.transform(embedding)

            >>> embedding.shape
            torch.Size([1000, 5])
            #
            >>> transformed_embedding.shape
            torch.Size([1000, 2])
        """
        features -= self.mean
        return torch.matmul(features, self.singular_vectors)

    def inverse_transform(self, features: torch.Tensor) -> torch.Tensor:
        """Inverses the transformed features.

        Args:
            features (torch.Tensor): Transformed features

        Returns:
            Inverse features

        Example:
            >>> inverse_embedding = pca.inverse_transform(transformed_embedding)
            >>> inverse_embedding.shape
            torch.Size([1000, 5])
        """
        return torch.matmul(features, self.singular_vectors.transpose(-2, -1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Transform the features.

        Args:
            features (torch.Tensor): Input features

        Returns:
            Transformed features

        Example:
            >>> pca(embedding).shape
            torch.Size([1000, 2])
        """
        return self.transform(features)
