"""This module comprises PatchCore Sampling Methods for the embedding.

- k Center Greedy Method
    Returns points that minimizes the maximum distance of any point to a center.
    . https://arxiv.org/abs/1708.00489
"""

from __future__ import annotations
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from anomalib.models.components.dimensionality_reduction import SparseRandomProjection


class KCenterGreedy:
    """Implements k-center-greedy method.

    Args:
        sampling_ratio (float): Ratio to choose coreset size from the embedding size.
    """

    def __init__(self, sampling_ratio: float) -> None:
        self.sampling_ratio = sampling_ratio
        self.features: Tensor
        self.min_distances: Tensor = None

    def reset_distances(self) -> None:
        """Reset minimum distances."""
        self.min_distances = None

    def update_distances(self, cluster_centers: list[int]) -> None:
        """Update min distances given cluster centers.

        Args:
            cluster_centers (list[int]): indices of cluster centers
        """

        if cluster_centers:
            centers = self.features[cluster_centers]

            distance = F.pairwise_distance(self.features, centers, p=2).reshape(-1, 1)

            if self.min_distances is None:
                self.min_distances = distance
            else:
                self.min_distances = torch.minimum(self.min_distances, distance)

    def get_new_idx(self) -> int:
        """Get index value of a sample.

        Based on minimum distance of the cluster

        Returns:
            int: Sample index
        """

        if isinstance(self.min_distances, Tensor):
            idx = int(torch.argmax(self.min_distances).item())
        else:
            raise ValueError(f"self.min_distances must be of type Tensor. Got {type(self.min_distances)}")

        return idx

    def select_coreset_idxs(self, embedding: Tensor, selected_idxs: list[int] | None = None) -> list[int]:
        """Greedily form a coreset to minimize the maximum distance of a cluster.

        Args:
            embedding: Embedding vector extracted from a CNN
            selected_idxs: index of samples already selected. Defaults to an empty set.

        Returns:
          indices of samples selected to minimize distance to cluster centers
        """

        if selected_idxs is None:
            selected_idxs = []

        if embedding.ndim != 2:
            self.features = embedding.reshape(embedding.shape[0], -1)
        else:
            self.features = embedding

        n_observations = self.features.shape[0]
        selected_coreset_idxs: list[int] = []
        idx = int(torch.randint(high=n_observations, size=(1,)).item())
        self.coreset_size = int(n_observations * self.sampling_ratio)

        for _ in tqdm(range(self.coreset_size), desc="Sampling Coreset"):
            self.update_distances(cluster_centers=[idx])
            idx = self.get_new_idx()
            if idx in selected_idxs:
                raise ValueError("New indices should not be in selected indices.")
            self.min_distances[idx] = 0
            selected_coreset_idxs.append(idx)

        del self.features
        return selected_coreset_idxs

    def sample_coreset(self, embedding: Tensor, selected_idxs: list[int] | None = None) -> Tensor:
        """Select coreset from the embedding.

        Args:
            embedding: Embedding vector extracted from a CNN
            selected_idxs: index of samples already selected. Defaults to an empty set.

        Returns:
            Tensor: Output coreset
        """
        idxs = self.select_coreset_idxs(embedding, selected_idxs)

        return idxs
