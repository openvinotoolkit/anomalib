"""Principle Component Analysis (PCA) with PyTorch."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import Tensor

from anomalib.models.components.base import DynamicBufferModule


class PCA(DynamicBufferModule):
    """Principle Component Analysis (PCA).

    Args:
        n_components (float): Number of components. Can be either integer number of components
          or a ratio between 0-1.
    """

    def __init__(self, n_components: int | float):
        super().__init__()
        self.n_components = n_components

        self.register_buffer("singular_vectors", Tensor())
        self.register_buffer("mean", Tensor())
        self.register_buffer("num_components", Tensor())

        self.singular_vectors: Tensor
        self.singular_values: Tensor
        self.mean: Tensor
        self.num_components: Tensor

    def fit(self, dataset: Tensor) -> None:
        """Fits the PCA model to the dataset.

        Args:
          dataset (Tensor): Input dataset to fit the model.
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

        self.num_components = Tensor([num_components])

        self.singular_vectors = v_h.transpose(-2, -1)[:, :num_components].float()
        self.singular_values = sig[:num_components].float()
        self.mean = mean

    def fit_transform(self, dataset: Tensor) -> Tensor:
        """Fit and transform PCA to dataset.

        Args:
          dataset (Tensor): Dataset to which the PCA if fit and transformed

        Returns:
          Transformed dataset
        """
        mean = dataset.mean(dim=0)
        dataset -= mean
        num_components = int(self.n_components)
        self.num_components = Tensor([num_components])

        v_h = torch.linalg.svd(dataset)[-1]
        self.singular_vectors = v_h.transpose(-2, -1)[:, :num_components]
        self.mean = mean

        return torch.matmul(dataset, self.singular_vectors)

    def transform(self, features: Tensor) -> Tensor:
        """Transforms the features based on singular vectors calculated earlier.

        Args:
          features (Tensor): Input features

        Returns:
          Transformed features
        """

        features -= self.mean
        return torch.matmul(features, self.singular_vectors)

    def inverse_transform(self, features: Tensor) -> Tensor:
        """Inverses the transformed features.

        Args:
          features (Tensor): Transformed features

        Returns: Inverse features
        """
        inv_features = torch.matmul(features, self.singular_vectors.transpose(-2, -1))
        return inv_features

    def forward(self, features: Tensor) -> Tensor:
        """Transforms the features.

        Args:
          features (Tensor): Input features

        Returns:
          Transformed features
        """
        return self.transform(features)
