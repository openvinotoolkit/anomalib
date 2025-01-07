"""Loss function for the CFA (Coupled-hypersphere-based Feature Adaptation) model.

This module implements the loss function used to train the CFA model for anomaly
detection. The loss consists of two components:
    1. Attraction loss that pulls normal samples inside a hypersphere
    2. Repulsion loss that pushes anomalous samples outside the hypersphere

Example:
    >>> import torch
    >>> from anomalib.models.image.cfa.loss import CfaLoss
    >>> # Initialize loss function
    >>> loss_fn = CfaLoss(
    ...     num_nearest_neighbors=3,
    ...     num_hard_negative_features=3,
    ...     radius=0.5
    ... )
    >>> # Compute loss on distance tensor
    >>> distance = torch.randn(2, 1024, 1)  # batch x pixels x 1
    >>> loss = loss_fn(distance)
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn


class CfaLoss(nn.Module):
    """Loss function for the CFA model.

    The loss encourages normal samples to lie within a hypersphere while pushing
    anomalous samples outside. It uses k-nearest neighbors to identify the closest
    samples and hard negative mining to find challenging anomalous examples.

    Args:
        num_nearest_neighbors (int): Number of nearest neighbors to consider for
            the attraction loss component.
        num_hard_negative_features (int): Number of hard negative features to use
            for the repulsion loss component.
        radius (float): Initial radius of the hypersphere that defines the
            decision boundary between normal and anomalous samples.

    Example:
        >>> loss_fn = CfaLoss(
        ...     num_nearest_neighbors=3,
        ...     num_hard_negative_features=3,
        ...     radius=0.5
        ... )
        >>> distance = torch.randn(2, 1024, 1)  # batch x pixels x 1
        >>> loss = loss_fn(distance)
    """

    def __init__(self, num_nearest_neighbors: int, num_hard_negative_features: int, radius: float) -> None:
        super().__init__()
        self.num_nearest_neighbors = num_nearest_neighbors
        self.num_hard_negative_features = num_hard_negative_features
        self.radius = torch.ones(1, requires_grad=True) * radius

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Compute the CFA loss given distance features.

        The loss has two components:
            1. Attraction loss (`l_att`): Encourages normal samples to lie within
               the hypersphere by penalizing distances greater than `radius`.
            2. Repulsion loss (`l_rep`): Pushes anomalous samples outside the
               hypersphere by penalizing distances less than `radius + margin`.

        Args:
            distance (torch.Tensor): Distance tensor of shape
                ``(batch_size, num_pixels, 1)`` computed using target-oriented
                features.

        Returns:
            torch.Tensor: Scalar loss value combining attraction and repulsion
                components.
        """
        num_neighbors = self.num_nearest_neighbors + self.num_hard_negative_features
        distance = distance.topk(num_neighbors, largest=False).values  # noqa: PD011

        score = distance[:, :, : self.num_nearest_neighbors] - (self.radius**2).to(distance.device)
        l_att = torch.mean(torch.max(torch.zeros_like(score), score))

        score = (self.radius**2).to(distance.device) - distance[:, :, self.num_hard_negative_features :]
        l_rep = torch.mean(torch.max(torch.zeros_like(score), score - 0.1))

        return (l_att + l_rep) * 1000
