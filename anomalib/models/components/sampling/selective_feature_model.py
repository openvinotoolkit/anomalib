"""Selective Feature Model."""

from typing import List

import numpy as np
import torch
from torch import Tensor, nn

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


class SelectiveFeatureModel(nn.Module):
    """Selective Feature Model.

    Args:
       feature_percentage (float): Percentage of features to keep.
    """

    def __init__(self, feature_percentage: float):
        super().__init__()

        # self.register_buffer("feature_stat", torch.zeros(n_features, n_patches))

        self.feature_percentage = feature_percentage

    def forward(self, max_activation_val: Tensor, class_labels: List[str]):
        """Store top `feature_percentage` features.

        Args:
          max_activation_val (Tensor): Max activation values of embeddings.
          class_labels (List[str]):  Corresponding sub-class labels.
        """
        class_names = np.unique(class_labels)

        for class_name in class_names:
            self.register_buffer(class_name, Tensor())
            setattr(self, class_name, Tensor())
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
            self.register_buffer(class_name, Tensor())
            setattr(self, class_name, torch.cat((sorted_idx.unsqueeze(0), sorted_idx_normalized.unsqueeze(0))))

    def fit(self, max_activation_val: Tensor, class_labels: List[str]):
        """Store top `feature_percentage` features.

        Args:
            max_activation_val (Tensor): Max activation values of embeddings.
            class_labels (List[str]):  Corresponding sub-class labels.
        """
        self.forward(max_activation_val, class_labels)
