"""KMeans clustering algorithm implementation using PyTorch."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch


class KMeans:
    """Initialize the KMeans object.

    Args:
        n_clusters (int): The number of clusters to create.
        max_iter (int, optional)): The maximum number of iterations to run the algorithm. Defaults to 10.
    """

    def __init__(self, n_clusters: int, max_iter: int = 10) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit the K-means algorithm to the input data.

        Args:
            inputs (torch.Tensor): Input data of shape (batch_size, n_features).

        Returns:
            tuple: A tuple containing the labels of the input data with respect to the identified clusters
            and the cluster centers themselves. The labels have a shape of (batch_size,) and the
            cluster centers have a shape of (n_clusters, n_features).

        Raises:
            ValueError: If the number of clusters is less than or equal to 0.
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

            # Update the centroids to be the mean of the data points assigned to them
            for j in range(self.n_clusters):
                mask = self.labels_ == j
                if mask.any():
                    self.cluster_centers_[j] = inputs[mask].mean(dim=0)
        # this line returns labels and centoids of the results
        return self.labels_, self.cluster_centers_

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict the labels of input data based on the fitted model.

        Args:
            inputs (torch.Tensor): Input data of shape (batch_size, n_features).

        Returns:
            torch.Tensor: The predicted labels of the input data with respect to the identified clusters.

        Raises:
            AttributeError: If the KMeans object has not been fitted to input data.
        """
        distances = torch.cdist(inputs, self.cluster_centers_)
        return torch.argmin(distances, dim=1)
