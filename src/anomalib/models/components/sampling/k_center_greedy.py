# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""k-Center Greedy Method.

Returns points that minimizes the maximum distance of any point to a center.

Reference:
    - https://arxiv.org/abs/1708.00489
"""

import torch
from tqdm import tqdm

from anomalib.models.components.dimensionality_reduction import SparseRandomProjection


class KCenterGreedy:
    """k-center-greedy method for coreset selection.

    This class implements the k-center-greedy method to select a coreset from an
    embedding space. The method aims to minimize the maximum distance between any
    point and its nearest center.

    Args:
        embedding (torch.Tensor): Embedding tensor extracted from a CNN.
        sampling_ratio (float): Ratio to determine coreset size from embedding size.

    Attributes:
        embedding (torch.Tensor): Input embedding tensor.
        coreset_size (int): Size of the coreset to be selected.
        model (SparseRandomProjection): Dimensionality reduction model.
        features (torch.Tensor): Transformed features after dimensionality reduction.
        min_distances (torch.Tensor): Minimum distances to cluster centers.
        n_observations (int): Number of observations in the embedding.

    Example:
        >>> import torch
        >>> embedding = torch.randn(219520, 1536)
        >>> sampler = KCenterGreedy(embedding=embedding, sampling_ratio=0.001)
        >>> sampled_idxs = sampler.select_coreset_idxs()
        >>> coreset = embedding[sampled_idxs]
        >>> coreset.shape
        torch.Size([219, 1536])
    """

    def __init__(self, embedding: torch.Tensor, sampling_ratio: float) -> None:
        self.embedding = embedding
        n = embedding.shape[0]
        coreset_size = int(n * sampling_ratio)
        if coreset_size <= 0:
            msg = (
                f"coreset_size must be a positive integer, got {coreset_size}. "
                f"Check sampling_ratio ({sampling_ratio}) and embedding size ({n})."
            )
            raise ValueError(msg)
        self.coreset_size = coreset_size
        self.model = SparseRandomProjection(eps=0.9)

        self.features: torch.Tensor
        self.min_distances: torch.Tensor = None
        self.n_observations = self.embedding.shape[0]

    def reset_distances(self) -> None:
        """Reset minimum distances to None."""
        self.min_distances = None

    def update_distances(self, cluster_center: int | torch.Tensor | None) -> None:
        """Update minimum distances given a single cluster center.

        Args:
            cluster_center (int | torch.Tensor | None): Index of a single cluster center.
                Can be an int, a 0-d tensor, or a 1-d tensor with shape [1].

        Note:
            This method is optimized for single-center updates. Passing multiple
            indices may result in incorrect behavior or runtime errors.
        """
        if cluster_center is not None:
            center = self.features[cluster_center]

            # Ensure center is a 1-d tensor for broadcasting
            center = center.squeeze()
            # Using torch.linalg.norm is faster than torch.Functional.pairwise_distance on
            # both CUDA and Intel-based hardware.
            distances = torch.linalg.norm(self.features - center, ord=2, dim=1, keepdim=True)
            # Synchronize on XPU to ensure accurate progress bar display.
            if distances.device.type == "xpu":
                torch.xpu.synchronize()

            if self.min_distances is None:
                self.min_distances = distances
            else:
                self.min_distances = torch.minimum(self.min_distances, distances)

    def get_new_idx(self) -> torch.Tensor:
        """Get index of the next sample based on maximum minimum distance.

        Returns:
            torch.Tensor: Index of the selected sample (tensor, not converted to int).

        Raises:
            TypeError: If `self.min_distances` is not a torch.Tensor.
        """
        if isinstance(self.min_distances, torch.Tensor):
            _, idx = torch.max(self.min_distances.squeeze(1), dim=0)
        else:
            msg = f"self.min_distances must be of type Tensor. Got {type(self.min_distances)}"
            raise TypeError(msg)
        return idx

    def select_coreset_idxs(self) -> list[int]:
        """Greedily select coreset indices to minimize maximum distance to centers.

        The algorithm iteratively selects points that are farthest from the already
        selected centers, starting from a random initial point.

        Returns:
            list[int]: Indices of samples selected to form the coreset.
        """
        if self.embedding.ndim == 2:
            self.model.fit(self.embedding)
            self.features = self.model.transform(self.embedding)
            self.reset_distances()
        else:
            self.features = self.embedding.reshape(self.embedding.shape[0], -1)
            self.reset_distances()

        # random starting point — include it in the coreset so it is not
        # merely a "virtual center" that influences distances but is absent
        # from the final memory bank (see gh-3459).
        idx = torch.randint(high=self.n_observations, size=(1,), device=self.features.device).squeeze()

        selected_coreset_idxs: list[torch.Tensor] = [idx]
        for _ in tqdm(range(self.coreset_size - 1), desc="Selecting Coreset Indices."):
            self.update_distances(cluster_center=idx)
            # Exclude selected centers even when all remaining distances are zero.
            # Subsequent minimum updates preserve this sentinel.
            self.min_distances.scatter_(0, idx.unsqueeze(0).unsqueeze(1), -torch.inf)
            idx = self.get_new_idx()
            selected_coreset_idxs.append(idx)

        return torch.stack(selected_coreset_idxs).cpu().tolist()

    def sample_coreset(self) -> torch.Tensor:
        """Select coreset from the embedding.

        Returns:
            torch.Tensor: Selected coreset.

        Example:
            >>> import torch
            >>> embedding = torch.randn(219520, 1536)
            >>> sampler = KCenterGreedy(embedding=embedding, sampling_ratio=0.001)
            >>> coreset = sampler.sample_coreset()
            >>> coreset.shape
            torch.Size([219, 1536])
        """
        idxs = self.select_coreset_idxs()
        return self.embedding[idxs]
