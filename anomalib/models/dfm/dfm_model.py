"""Normality model of DFKDE."""

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

import math

import torch
from torch import Tensor, nn

from anomalib.core.model.dynamic_module import DynamicBufferModule
from anomalib.core.model.pca import PCA


class SingleClassGaussian(DynamicBufferModule):
    """Model Gaussian distribution over a set of points."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean_vec", Tensor())
        self.register_buffer("u_mat", Tensor())
        self.register_buffer("sigma_mat", Tensor())

        self.mean_vec: Tensor
        self.u_mat: Tensor
        self.sigma_mat: Tensor

    def fit(self, dataset: Tensor) -> None:
        """Fit a Gaussian model to dataset X.

        Covariance matrix is not calculated directly using:
        ``C = X.X^T``
        Instead, it is represented in terms of the Singular Value Decomposition of X:
        ``X = U.S.V^T``
        Hence,
        ``C = U.S^2.U^T``
        This simplifies the calculation of the log-likelihood without requiring full matrix inversion.

        Args:
            dataset (Tensor): Input dataset to fit the model.
        """

        num_samples = dataset.shape[1]
        self.mean_vec = torch.mean(dataset, dim=1)
        data_centered = (dataset - self.mean_vec.reshape(-1, 1)) / math.sqrt(num_samples)
        self.u_mat, self.sigma_mat, _ = torch.linalg.svd(data_centered, full_matrices=False)

    def score_samples(self, features: Tensor) -> Tensor:
        """Compute the NLL (negative log likelihood) scores.

        Args:
            features (Tensor): semantic features on which density modeling is performed.

        Returns:
            nll (Tensor): Torch tensor of scores
        """
        features_transformed = torch.matmul(features - self.mean_vec, self.u_mat / self.sigma_mat)
        nll = torch.sum(features_transformed * features_transformed, dim=1) + 2 * torch.sum(torch.log(self.sigma_mat))
        return nll

    def forward(self, dataset: Tensor) -> None:
        """Provides the same functionality as `fit`.

        Transforms the input dataset based on singular values calculated earlier.

        Args:
            dataset (Tensor): Input dataset
        """
        self.fit(dataset)


class DFMModel(nn.Module):
    """Model for the DFM algorithm.

    Args:
        n_comps (float, optional): Ratio from which number of components for PCA are calculated. Defaults to 0.97.
        score_type (str, optional): Scoring type. Options are `fre` and `nll`. Defaults to "fre".
    """

    def __init__(self, n_comps: float = 0.97, score_type: str = "fre"):
        super().__init__()
        self.n_components = n_comps
        self.pca_model = PCA(n_components=self.n_components)
        self.gaussian_model = SingleClassGaussian()
        self.score_type = score_type

    def fit(self, dataset: Tensor) -> None:
        """Fit a pca transformation and a Gaussian model to dataset.

        Args:
            dataset (Tensor): Input dataset to fit the model.
        """

        self.pca_model.fit(dataset)
        features_reduced = self.pca_model.transform(dataset)
        self.gaussian_model.fit(features_reduced.T)

    def score(self, features: Tensor) -> Tensor:
        """Compute scores.

        Scores are either PCA-based feature reconstruction error (FRE) scores or
        the Gaussian density-based NLL scores

        Args:
            features (torch.Tensor): semantic features on which PCA and density modeling is performed.

        Returns:
            score (Tensor): numpy array of scores
        """
        feats_projected = self.pca_model.transform(features)
        if self.score_type == "nll":
            score = self.gaussian_model.score_samples(feats_projected)
        elif self.score_type == "fre":
            feats_reconstructed = self.pca_model.inverse_transform(feats_projected)
            score = torch.sum(torch.square(features - feats_reconstructed), dim=1)
        else:
            raise ValueError(f"unsupported score type: {self.score_type}")

        return score

    def forward(self, dataset: Tensor) -> None:
        """Provides the same functionality as `fit`.

        Transforms the input dataset based on singular values calculated earlier.

        Args:
            dataset (Tensor): Input dataset
        """
        self.fit(dataset)
