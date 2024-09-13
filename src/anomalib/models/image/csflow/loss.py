"""Loss function for the CS-Flow Model Implementation."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn


class CsFlowLoss(nn.Module):
    """Loss function for the CS-Flow Model Implementation."""

    @staticmethod
    def forward(z_dist: torch.Tensor, jacobians: torch.Tensor) -> torch.Tensor:
        """Compute the loss CS-Flow.

        Args:
            z_dist (torch.Tensor): Latent space image mappings from NF.
            jacobians (torch.Tensor): Jacobians of the distribution

        Returns:
            Loss value
        """
        z_dist = torch.cat([z_dist[i].reshape(z_dist[i].shape[0], -1) for i in range(len(z_dist))], dim=1)
        return torch.mean(0.5 * torch.sum(z_dist**2, dim=(1,)) - jacobians) / z_dist.shape[1]
