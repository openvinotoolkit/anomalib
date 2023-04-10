#KMeans clustering algorithm implementation in PyTorch

import torch

class KMeans:
    def __init__(self, n_clusters: int, max_iter:int = 10):
        """
        Initializes the KMeans object.

        Parameters:
            n_clusters: The number of clusters to create.
            max_iter: The maximum number of iterations to run the algorithm for.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        """
        Runs the k-means algorithm on input data X.

        Parameters:
            X: A tensor of shape (N, D) containing the input data.
            N is the number of data points 
            D is the dimensionality of the data points.
        """
        N, D = X.shape

        # Initialize centroids randomly from the data points
        centroid_indices = torch.randint(0, N, (self.n_clusters,))
        self.cluster_centers_ = X[centroid_indices]

        # Run the k-means algorithm for max_iter iterations
        for i in range(self.max_iter):
            # Compute the distance between each data point and each centroid
            distances = torch.cdist(X, self.cluster_centers_)

            # Assign each data point to the closest centroid
            self.labels_ = torch.argmin(distances, dim=1)

            # Update the centroids to be the mean of the data points assigned to them
            for j in range(self.n_clusters):
                mask = self.labels_ == j
                if mask.any():
                    self.cluster_centers_[j] = X[mask].mean(dim=0)        
        #thise line returns labels and centoids of the results,          
        return self.labels_, self.cluster_centers_

    def predict(self, X):
        """
        Assigns each data point in X to its closest centroid.

        Parameters:
            X: A tensor of shape (N, D) containing the input data.

        Returns:
            A tensor of shape (N,) containing the index of the closest centroid for each data point.
        """
        distances = torch.cdist(X, self.cluster_centers_)
        return torch.argmin(distances, dim=1)
