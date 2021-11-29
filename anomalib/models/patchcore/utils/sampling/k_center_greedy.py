"""This module comprises PatchCore Sampling Methods for the embedding.

- k Center Greedy Method
    Returns points that minimizes the maximum distance of any point to a center.
    . https://arxiv.org/abs/1708.00489
"""

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .random_projection import SparseRandomProjection


class KCenterGreedy:
    """Implements k-center-greedy method.

    Args:
        model: model with scikit-like API with decision_function. Defaults to SparseRandomProjection.
        embedding (Tensor): Embedding vector extracted from a CNN
        sampling_ratio (float): Ratio to choose coreset size from the embedding size.

    Example:
        >>> embedding.shape
        torch.Size([219520, 1536])
        >>> sampler = KCenterGreedy(embedding=embedding)
        >>> sampled_idxs = sampler.select_coreset_idxs()
        >>> coreset = embedding[sampled_idxs]
        >>> coreset.shape
        torch.Size([219, 1536])
    """

    def __init__(self, model: SparseRandomProjection, embedding: Tensor, sampling_ratio: float) -> None:
        self.model = model
        self.embedding = embedding
        self.coreset_size = int(embedding.shape[0] * sampling_ratio)

        self.features: Tensor
        self.min_distances: Optional[Tensor] = None
        self.n_observations = self.embedding.shape[0]
        self.already_selected_idxs: List[int] = []

    def reset_distances(self) -> None:
        """Reset minimum distances."""
        self.min_distances = None

    def get_new_cluster_centers(self, cluster_centers: List[int]) -> List[int]:
        """Get new cluster center indexes from the list of cluster indexes.

        Args:
            cluster_centers (List[int]): List of cluster center indexes.

        Returns:
            List[int]: List of new cluster center indexes.
        """
        return [d for d in cluster_centers if d not in self.already_selected_idxs]

    def update_distances(self, cluster_centers: List[int]) -> None:
        """Update min distances given cluster centers.

        Args:
            cluster_centers (List[int]): indices of cluster centers
        """

        if cluster_centers:
            cluster_centers = self.get_new_cluster_centers(cluster_centers)
            centers = self.features[cluster_centers]

            distance = F.pairwise_distance(self.features, centers, p=2).reshape(-1, 1)

            if self.min_distances is None:
                self.min_distances = torch.min(distance, dim=1).values.reshape(-1, 1)
            else:
                self.min_distances = torch.minimum(self.min_distances, distance)

    def get_new_idx(self) -> int:
        """Get index value of a sample.

        Based on (i) either minimum distance of the cluster or (ii) random subsampling from the embedding.

        Returns:
            int: Sample index
        """

        if self.already_selected_idxs is None or len(self.already_selected_idxs) == 0:
            # Initialize centers with a randomly selected datapoint
            idx = int(torch.randint(high=self.n_observations, size=(1,)).item())
        else:
            if isinstance(self.min_distances, Tensor):
                idx = int(torch.argmax(self.min_distances).item())
            else:
                raise ValueError(f"self.min_distances must be of type Tensor. Got {type(self.min_distances)}")

        return idx

    def select_coreset_idxs(self, selected_idxs: Optional[List[int]] = None) -> List[int]:
        """Greedily form a coreset to minimize the maximum distance of a cluster.

        Args:
            selected_idxs: index of samples already selected. Defaults to an empty set.

        Returns:
          indices of samples selected to minimize distance to cluster centers
        """

        if selected_idxs is None:
            selected_idxs = []

        if self.embedding.ndim == 2:
            self.features = self.model.transform(self.embedding)
            self.reset_distances()
        else:
            self.features = self.embedding.reshape(self.embedding.shape[0], -1)
            self.update_distances(cluster_centers=selected_idxs)

        selected_coreset_idxs: List[int] = []
        for _ in range(self.coreset_size):
            idx = self.get_new_idx()
            if idx in selected_idxs:
                raise ValueError("New indices should not be in selected indices.")

            self.update_distances(cluster_centers=[idx])
            selected_coreset_idxs.append(idx)

        self.already_selected_idxs = selected_idxs

        return selected_coreset_idxs

    def sample_coreset(self, selected_idxs: Optional[List[int]] = None) -> Tensor:
        """Select coreset from the embedding.

        Args:
            selected_idxs: index of samples already selected. Defaults to an empty set.

        Returns:
            Tensor: Output coreset

        Example:
            >>> embedding.shape
            torch.Size([219520, 1536])
            >>> sampler = KCenterGreedy(...)
            >>> coreset = sampler.sample_coreset()
            >>> coreset.shape
            torch.Size([219, 1536])
        """

        idxs = self.select_coreset_idxs(selected_idxs)
        coreset = self.embedding[idxs]

        return coreset
