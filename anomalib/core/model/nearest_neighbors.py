"""Nearest neighbours using pytorch's methods"""
from typing import Tuple

import torch
from torch import Tensor

from anomalib.core.model.dynamic_module import DynamicBufferModule


class NearestNeighbors(DynamicBufferModule):
    """Nearest Neighbours using brute force method and euclidean norm

    Args:
        n_neighbors (int): Number of neighbors to look at
    """

    def __init__(self, n_neighbors: int):
        super().__init__()
        self.n_neighbors = n_neighbors

        self.register_buffer("_fit_x", Tensor())
        self._fit_x: Tensor

    def fit(self, train_features: Tensor):
        """Saves the train features for NN search later

        Args:
            train_features (Tensor): Training data
        """
        self._fit_x = train_features

    def kneighbors(self, test_features: Tensor) -> Tuple[Tensor, Tensor]:
        """Return k-nearest neighbors. It is calculated based on bruteforce method.

        Args:
            test_features (Tensor): test data

        Returns:
            Tuple[Tensor, Tensor]: distances, indices
        """
        distances = torch.cdist(test_features, self._fit_x, p=2.0)  # euclidean norm
        return distances.topk(k=self.n_neighbors, largest=False, dim=1)
