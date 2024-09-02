"""Loss function for the UFlow Model Implementation."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor, nn


class UFlowLoss(nn.Module):
    """UFlow Loss."""

    @staticmethod
    def forward(hidden_variables: list[Tensor], jacobians: list[Tensor]) -> Tensor:
        """Calculate the UFlow loss.

        Args:
            hidden_variables (list[Tensor]): Hidden variables from the fastflow model. f: X -> Z
            jacobians (list[Tensor]): Log of the jacobian determinants from the fastflow model.

        Returns:
            Tensor: UFlow loss computed based on the hidden variables and the log of the Jacobians.
        """
        lpz = torch.sum(torch.stack([0.5 * torch.sum(z_i**2, dim=(1, 2, 3)) for z_i in hidden_variables], dim=0))
        return torch.mean(lpz - jacobians)
