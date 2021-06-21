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

        self.register_buffer("singular_vectors", torch.Tensor())
        self.register_buffer("mean", torch.Tensor())

        self.singular_vectors = torch.Tensor()
        self.mean = torch.Tensor()

    def fit_transform(self, dataset: torch.Tensor) -> torch.Tensor:
        """

        Args:
          dataset: torch.Tensor:

        Returns:

        """
        mean = dataset.mean(dim=0)
        dataset -= mean

        self.singular_vectors = torch.svd(dataset)[-1]
        self.mean = mean

        return torch.matmul(dataset, self.singular_vectors[:, : self.n_components])

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """

        Args:
          features: torch.Tensor:

        Returns:

        """
        features -= self.mean
        return torch.matmul(features, self.singular_vectors[:, : self.n_components])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """

        Args:
          features: torch.Tensor:

        Returns:

        """
        return self.transform(features)
