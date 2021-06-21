"""
Principle Component Analysis (PCA) with PyTorch
"""

import torch

from anomalib.core.model.dynamic_module import DynamicBufferModule


class PCA(DynamicBufferModule):
    """
    Principle Component Analysis (PCA)
    """

    def __init__(self, n_components: int):
        super().__init__()
        self.n_components = n_components

        self.register_buffer("singularity_vector", torch.Tensor())
        self.register_buffer("mean", torch.Tensor())

        self.singularity_vector = torch.Tensor()
        self.mean = torch.Tensor()

    def fit_transform(self, dataset: torch.Tensor) -> torch.Tensor:
        """

        Args:
          dataset: torch.Tensor:

        Returns:

        """
        mean = dataset.mean(dim=0)
        dataset -= mean

        self.singularity_vector = torch.svd(dataset)[-1]
        self.mean = mean

        return torch.matmul(dataset, self.singularity_vector[:, : self.n_components])

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """

        Args:
          features: torch.Tensor:

        Returns:

        """
        features -= self.mean
        return torch.matmul(features, self.singularity_vector[:, : self.n_components])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """

        Args:
          features: torch.Tensor:

        Returns:

        """
        return self.transform(features)
