from typing import List

import numpy as np
import torch
from torch import Tensor, nn


class SelectiveFeatureModel(nn.Module):
    """Selective Feature Modelling."""

    def __init__(self, feature_percentage: float):
        super().__init__()

        # self.register_buffer("feature_stat", torch.zeros(n_features, n_patches))

        self.feature_percentage = feature_percentage
        self.class_stats = {}

    def forward(self, max_activation_val: Tensor, class_labels: List[str]):
        """Calculate multivariate Gaussian distribution.
        Args:
          embedding (Tensor): CNN features whose dimensionality is reduced via either random sampling or PCA.
        """

        class_names = np.unique(class_labels)
        # print(class_labels)
        # print(max_activation_val.shape)

        for class_name in class_names:
            # print(class_name)
            self.register_buffer(class_name, Tensor())
            setattr(self, class_name, Tensor())
            # print(max_activation_val.shape)
            # print(np.where(class_labels == class_name))
            class_max_activations = max_activation_val[class_labels == class_name]
            # sorted values and idx for entire feature set
            max_val, max_idx = torch.sort(class_max_activations, descending=True)
            reduced_range = int(max_val.shape[1] * self.feature_percentage)
            # indexes of top 10% FEATURES HAVING MAX VALUE
            top_max_idx = max_idx[:, 0:reduced_range]
            # out of sorted top 10, what features are affiliated the most
            idx, repetitions = torch.unique(top_max_idx, return_counts=True)
            sorted_repetition, sorted_repetition_idx = torch.sort(repetitions, descending=True)
            sorted_idx = idx[sorted_repetition_idx]

            sorted_idx_normalized = sorted_repetition / class_max_activations.shape[0]
            sorted_idx_normalized = sorted_idx_normalized / sorted_idx_normalized.sum()
            # print(torch.cat((sorted_idx.unsqueeze(0), sorted_idx_normalized.unsqueeze(0))))
            self.register_buffer(class_name, Tensor())
            setattr(self, class_name, torch.cat((sorted_idx.unsqueeze(0), sorted_idx_normalized.unsqueeze(0))))

    def fit(self, max_val_features: Tensor, class_labels: List[str]):
        """Fit multi-variate gaussian distribution to the input embedding.
        Args:
            embedding (Tensor): Embedding vector extracted from CNN.
        Returns:
            Mean and the covariance of the embedding.
        """
        self.forward(max_val_features, class_labels)
