"""Loss function for the FastFlow Model Implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import Tensor, nn


class FastflowLoss(nn.Module):
    """FastFlow Loss."""

    def forward(self, hidden_variables: list[Tensor], jacobians: list[Tensor]) -> Tensor:
        """Calculate the Fastflow loss.

        Args:
            hidden_variables (list[Tensor]): Hidden variables from the fastflow model. f: X -> Z
            jacobians (list[Tensor]): Log of the jacobian determinants from the fastflow model.

        Returns:
            Tensor: Fastflow loss computed based on the hidden variables and the log of the Jacobians.
        """
        loss = torch.tensor(0.0, device=hidden_variables[0].device)  # pylint: disable=not-callable
        for hidden_variable, jacobian in zip(hidden_variables, jacobians):
            loss += torch.mean(0.5 * torch.sum(hidden_variable**2, dim=(1, 2, 3)) - jacobian)
        return loss
