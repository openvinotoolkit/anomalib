"""
Normality model of DFKDE
"""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import numpy as np
import torch
from sklearn.decomposition import PCA


class SingleclassGaussian:
    """
    Model Gaussian distribution over a set of points
    """

    def __init__(self):
        self.mean_vec = None
        self.u_mat = None
        self.sigma_mat = None

    def fit(self, dataset):
        """
        Fit a Gaussian model to dataset X.
        Covariance matrix is not calculated directly using:
            C = X.X^T
        Instead, it is represented in terms of the Singular Value Decomposition of X:
            X = U.S.V^T
        Hence,
            C = U.S^2.U^T
        This simplifies the calculation of the log-likelihood without requiring full matrix inversion.

        Args:
            dataset: Input dataset to fit the model.
            dataset: torch.Tensor:

        Returns:

        """

        num_samples = dataset.shape[1]
        self.mean_vec = torch.mean(dataset, dim=1)
        data_centered = (dataset - self.mean_vec.reshape(-1, 1)) / torch.sqrt(torch.Tensor([num_samples]))
        self.u_mat, self.sigma_mat, _ = torch.linalg.svd(data_centered, full_matrices=False)

    def score_samples(self, features):
        """
        Compute the NLL (negative log likelihood) scores

        Args:
            x: semantic features on which density modeling is performed.

        Returns:
            nll: numpy array of scores

        """
        features_transformed = torch.matmul(features - self.mean_vec, self.u_mat / self.sigma_mat)
        nll = torch.sum(features_transformed * features_transformed, dim=1) + 2 * np.sum(np.log(self.sigma_mat))
        return nll


class DFMModel:
    """
    Model for the DFM algorithm
    """

    def __init__(self, n_comps: float = 0.97, score_type: str = "fre"):
        super().__init__()
        self.n_components = n_comps
        self.pca_model = PCA(n_components=self.n_components)
        self.gaussian_model = SingleclassGaussian()
        self.score_type = score_type

    def fit(self, dataset: torch.Tensor):
        """
        Fit a pca transformation and a Gaussian model to dataset

        Args:
            dataset: Input dataset to fit the model.
            dataset: torch.Tensor:

        Returns:

        """

        selected_features = dataset.cpu().numpy()
        self.pca_model.fit(selected_features)
        features_reduced = torch.Tensor(self.pca_model.transform(selected_features))
        self.gaussian_model.fit(features_reduced.T)

    def score(self, sem_feats: torch.Tensor) -> np.array:
        """
        Compute the PCA-based feature reconstruction error (FRE) scores and
        the Gaussian density-based NLL scores

        Args:
            sem_feats: semantic features on which PCA and density modeling is performed.

        Returns:
            score: numpy array of scores

        """
        feats_orig = sem_feats.cpu().numpy()
        feats_projected = self.pca_model.transform(feats_orig)
        if self.score_type == "nll":
            score = self.gaussian_model.score_samples(feats_projected)
        elif self.score_type == "fre":
            feats_reconstructed = self.pca_model.inverse_transform(feats_projected)
            score = np.sum(np.square(feats_orig - feats_reconstructed), axis=1)
        return score
