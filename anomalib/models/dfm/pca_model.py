"""
Normality model of DFKDE
"""

import torch
import torch.nn as nn
import numpy as np

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

        """


        selected_features = dataset
        self.pca_model.fit(selected_features.cpu().numpy())


    def score(self, sem_feats: torch.Tensor) -> np.array:
        """
        Compute the PCA scores

        Args:
            sem_feats: semantic features on which PCA and density modeling is performed.

        Returns: 
            score: numpy array of scores

        """
        feats_orig = sem_feats.cpu().numpy()
        feats_projected = self.pca_model.transform(feats_orig)
        feats_reconstructed = self.pca_model.inverse_transform(feats_projected)
        score = np.sum(np.square(feats_orig - feats_reconstructed), axis=1)
        return score

