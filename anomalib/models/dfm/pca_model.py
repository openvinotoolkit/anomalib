"""
Normality model of DFKDE
"""

import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from anomalib.core.model.kde import GaussianKDE
from sklearn.decomposition import PCA


class PCAModel(nn.Module):
    """
    PCA Model for the DFM algorithm
    """

    def __init__(
        self,
        n_comps: float = 0.97,
    ):
        super().__init__()
        self.n_components = n_comps
        self.pca_model = PCA(n_components=self.n_components)

    def fit(self, dataset: torch.Tensor):
        """
        Fit a pca model to dataset

        Args:
            dataset: Input dataset to fit the model.
            dataset: torch.Tensor:

        Returns:
            Boolean confirming whether the training is successful.

        """


        selected_features = dataset
        self.pca_model.fit(selected_features.cpu().numpy())

        return True


    def score(self, sem_feats: torch.Tensor) -> torch.Tensor:
        """
        Compute the PCA scores

        Args:
            sem_feats:

        Returns:

        """
        feats_orig = sem_feats.cpu().numpy()
        feats_projected = self.pca_model.transform(feats_orig)
        feats_reconstructed = self.pca_model.inverse_transform(feats_projected)
        score = np.sum(np.square(feats_orig - feats_reconstructed), axis=1)
        return score

