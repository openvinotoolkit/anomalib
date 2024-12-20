"""PyTorch implementation of K-means clustering algorithm.

This module provides a PyTorch-based implementation of the K-means clustering
algorithm for partitioning data into ``k`` distinct clusters.

Example:
    >>> import torch
    >>> from anomalib.models.components.cluster import KMeans
    >>> # Create synthetic data
    >>> data = torch.tensor([
    ...     [1.0, 2.0], [1.5, 1.8], [1.2, 2.2],  # Cluster 1
    ...     [4.0, 4.0], [4.2, 4.1], [3.8, 4.2],  # Cluster 2
    ... ])
    >>> # Initialize and fit KMeans
    >>> kmeans = KMeans(n_clusters=2)
    >>> labels, centers = kmeans.fit(data)
    >>> # Predict cluster for new points
    >>> new_points = torch.tensor([[1.1, 2.1], [4.0, 4.1]])
    >>> predictions = kmeans.predict(new_points)
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch


class KMeans:
    """K-means clustering algorithm implementation.

    Args:
        n_clusters (int): Number of clusters to partition the data into.
        max_iter (int, optional): Maximum number of iterations for the clustering
            algorithm. Defaults to 10.

    Attributes:
        cluster_centers_ (torch.Tensor): Coordinates of cluster centers after
            fitting. Shape: ``(n_clusters, n_features)``.
        labels_ (torch.Tensor): Cluster labels for the training data after
            fitting. Shape: ``(n_samples,)``.

    Example:
        >>> import torch
        >>> from anomalib.models.components.cluster import KMeans
        >>> kmeans = KMeans(n_clusters=3)
        >>> data = torch.randn(100, 5)  # 100 samples, 5 features
        >>> labels, centers = kmeans.fit(data)
        >>> print(f"Cluster assignments shape: {labels.shape}")
        >>> print(f"Cluster centers shape: {centers.shape}")
    """

    def __init__(self, n_clusters: int, max_iter: int = 10) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit the K-means algorithm to the input data.

        Args:
            inputs (torch.Tensor): Input data to cluster.
                Shape: ``(n_samples, n_features)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - labels: Cluster assignments for each input point.
                  Shape: ``(n_samples,)``
                - cluster_centers: Coordinates of the cluster centers.
                  Shape: ``(n_clusters, n_features)``

        Raises:
            ValueError: If ``n_clusters`` is less than or equal to 0.

        Example:
            >>> kmeans = KMeans(n_clusters=2)
            >>> data = torch.tensor([[1.0, 2.0], [4.0, 5.0], [1.2, 2.1]])
            >>> labels, centers = kmeans.fit(data)
            >>> print(f"Number of points in each cluster: {
            ...     [(labels == i).sum().item() for i in range(2)]
            ... }")
        """
        batch_size, _ = inputs.shape

        # Initialize centroids randomly from the data points
        centroid_indices = torch.randint(0, batch_size, (self.n_clusters,))
        self.cluster_centers_ = inputs[centroid_indices]

        # Run the k-means algorithm for max_iter iterations
        for _ in range(self.max_iter):
            # Compute the distance between each data point and each centroid
            distances = torch.cdist(inputs, self.cluster_centers_)

            # Assign each data point to the closest centroid
            self.labels_ = torch.argmin(distances, dim=1)

            # Update the centroids to be the mean of the data points assigned
            for j in range(self.n_clusters):
                mask = self.labels_ == j
                if mask.any():
                    self.cluster_centers_[j] = inputs[mask].mean(dim=0)

        return self.labels_, self.cluster_centers_

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict cluster labels for input data.

        Args:
            inputs (torch.Tensor): Input data to assign to clusters.
                Shape: ``(n_samples, n_features)``.

        Returns:
            torch.Tensor: Predicted cluster labels.
                Shape: ``(n_samples,)``.

        Raises:
            AttributeError: If called before fitting the model.

        Example:
            >>> kmeans = KMeans(n_clusters=2)
            >>> # First fit the model
            >>> train_data = torch.tensor([[1.0, 2.0], [4.0, 5.0]])
            >>> kmeans.fit(train_data)
            >>> # Then predict on new data
            >>> new_data = torch.tensor([[1.1, 2.1], [3.9, 4.8]])
            >>> predictions = kmeans.predict(new_data)
        """
        distances = torch.cdist(inputs, self.cluster_centers_)
        return torch.argmin(distances, dim=1)
