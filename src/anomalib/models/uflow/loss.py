"""Loss function for the UFlow Model Implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import Tensor, nn


class UFlowLoss(nn.Module):
    """UFlow Loss."""

    def forward(self, hidden_variables: list[Tensor], jacobians: list[Tensor]) -> Tensor:
        """Calculate the UFlow loss.

        Args:
            hidden_variables (list[Tensor]): Hidden variables from the fastflow model. f: X -> Z
            jacobians (list[Tensor]): Log of the jacobian determinants from the fastflow model.

        Returns:
            Tensor: UFlow loss computed based on the hidden variables and the log of the Jacobians.
        """
        lpz = torch.sum(torch.stack([0.5 * torch.sum(z_i**2, dim=(1, 2, 3)) for z_i in hidden_variables], dim=0))
        flow_loss = torch.mean(lpz - jacobians)
        return flow_loss
