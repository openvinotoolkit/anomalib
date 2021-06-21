"""
Gaussian Kernel Density Estimation
"""

import math
from typing import Optional

import torch

from anomalib.core.model.dynamic_module import DynamicBufferModule


class GaussianKDE(DynamicBufferModule):
    """
    Gaussian Kernel Density Estimation
    """

    def __init__(self, dataset: Optional[torch.Tensor] = None):
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
        """
        Get the KDE estimates from the feature map.

        Args:
          features: torch.Tensor: Feature map extracted from the CNN

        Returns: KDE Estimates

        """
        features = torch.matmul(features, self.bw_transform)

        estimate = torch.zeros(features.shape[0])
        for j in range(features.shape[0]):
            embedding = ((self.dataset - features[j]) ** 2).sum(dim=1)
            embedding = torch.exp(-embedding / 2) * self.norm
            estimate[j] = torch.mean(embedding)

        return estimate

    def fit(self, dataset: torch.Tensor):
        """
        Fit a KDE model to the input dataset.

        Args:
          dataset: torch.Tensor: Input dataset.

        Returns:

        """
        num_samples, dimension = dataset.shape

        # compute scott's bandwidth factor
        factor = num_samples ** (-1 / (dimension + 4))

        cov_mat = self.cov(dataset.T, bias=False)
        inv_cov_mat = torch.linalg.inv(cov_mat)
        inv_cov = inv_cov_mat / factor ** 2

        # transform data to account for bandwidth
        bw_transform = torch.linalg.cholesky(inv_cov)
        dataset = torch.matmul(dataset, bw_transform)

        #
        norm = torch.prod(torch.diag(bw_transform))
        norm *= math.pow((2 * math.pi), (-dimension / 2))

        self.bw_transform = bw_transform
        self.dataset = dataset
        self.norm = norm

    @staticmethod
    def cov(tensor: torch.Tensor, bias: Optional[bool] = False) -> torch.Tensor:
        """
        Calculate covariance matrix.

        Args:
            tensor: torch.Tensor: Input tensor from which covariance matrix is computed.
            bias: Optional[bool]:  (Default value = False)

        Returns:
            Output covariance matrix.

        """
        mean = torch.mean(tensor, dim=1)
        tensor -= mean[:, None]
        cov = torch.matmul(tensor, tensor.T) / (tensor.size(1) - int(not bias))
        return cov
