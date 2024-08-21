"""Loss function for the Cfa Model Implementation."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn


class CfaLoss(nn.Module):
    """Cfa Loss.

    Args:
        num_nearest_neighbors (int): Number of nearest neighbors.
        num_hard_negative_features (int): Number of hard negative features.
        radius (float): Radius of the hypersphere to search the soft boundary.
    """

    def __init__(self, num_nearest_neighbors: int, num_hard_negative_features: int, radius: float) -> None:
        super().__init__()
        self.num_nearest_neighbors = num_nearest_neighbors
        self.num_hard_negative_features = num_hard_negative_features
        self.radius = torch.ones(1, requires_grad=True) * radius

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Compute the CFA loss.

        Args:
            distance (torch.Tensor): Distance computed using target oriented features.

        Returns:
            Tensor: CFA loss.
        """
        num_neighbors = self.num_nearest_neighbors + self.num_hard_negative_features
        distance = distance.topk(num_neighbors, largest=False).values  # noqa: PD011

        score = distance[:, :, : self.num_nearest_neighbors] - (self.radius**2).to(distance.device)
        l_att = torch.mean(torch.max(torch.zeros_like(score), score))

        score = (self.radius**2).to(distance.device) - distance[:, :, self.num_hard_negative_features :]
        l_rep = torch.mean(torch.max(torch.zeros_like(score), score - 0.1))

        return (l_att + l_rep) * 1000
