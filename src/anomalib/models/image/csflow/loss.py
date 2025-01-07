"""Loss function for the CS-Flow Model Implementation.

This module implements the loss function used in the CS-Flow model for anomaly
detection. The loss combines the squared L2 norm of the latent space
representations with the log-determinant of the Jacobian from the normalizing
flows.

Example:
    >>> import torch
    >>> from anomalib.models.image.csflow.loss import CsFlowLoss
    >>> criterion = CsFlowLoss()
    >>> z_dist = [torch.randn(2, 64, 32, 32) for _ in range(3)]
    >>> jacobians = torch.randn(2)
    >>> loss = criterion(z_dist, jacobians)
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn


class CsFlowLoss(nn.Module):
    """Loss function for the CS-Flow model.

    The loss is computed as the mean of the squared L2 norm of the latent space
    representations minus the log-determinant of the Jacobian, normalized by the
    dimensionality of the latent space.
    """

    @staticmethod
    def forward(z_dist: list[torch.Tensor], jacobians: torch.Tensor) -> torch.Tensor:
        """Compute the CS-Flow loss.

        Args:
            z_dist (list[torch.Tensor]): List of latent space tensors from each
                scale of the normalizing flow. Each tensor has shape
                ``(batch_size, channels, height, width)``.
            jacobians (torch.Tensor): Log-determinant of the Jacobian matrices
                from the normalizing flows. Shape: ``(batch_size,)``.

        Returns:
            torch.Tensor: Scalar loss value averaged over the batch.

        Example:
            >>> z_dist = [torch.randn(2, 64, 32, 32) for _ in range(3)]
            >>> jacobians = torch.randn(2)
            >>> loss = CsFlowLoss.forward(z_dist, jacobians)
        """
        concatenated = torch.cat([z_dist[i].reshape(z_dist[i].shape[0], -1) for i in range(len(z_dist))], dim=1)
        return torch.mean(0.5 * torch.sum(concatenated**2, dim=(1,)) - jacobians) / concatenated.shape[1]
