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

        self.register_buffer("tensor_u", torch.Tensor())
        self.register_buffer("tensor_s", torch.Tensor())
        self.register_buffer("tensor_v", torch.Tensor())
        self.register_buffer("mean", torch.Tensor())

        self.tensor_u = torch.Tensor()
        self.tensor_s = torch.Tensor()
        self.tensor_v = torch.Tensor()
        self.mean = torch.Tensor()

    def fit_transform(self, dataset: torch.Tensor) -> torch.Tensor:
        """

        Args:
          dataset: torch.Tensor:

        Returns:

        """
        mean = dataset.mean(dim=0)
        dataset -= mean

        tensor_u, tensor_s, tensor_v = torch.svd(dataset)

        self.tensor_u = tensor_u
        self.tensor_s = tensor_s
        self.tensor_v = tensor_v
        self.mean = mean

        return torch.matmul(dataset, tensor_v[:, : self.n_components])

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """

        Args:
          features: torch.Tensor:

        Returns:

        """
        features -= self.mean
        return torch.matmul(features, self.tensor_v[:, : self.n_components])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """

        Args:
          features: torch.Tensor:

        Returns:

        """
        return self.transform(features)
