"""Loss function for the FastFlow Model Implementation.

This module implements the loss function used to train the FastFlow model. The loss is
computed based on the hidden variables and Jacobian determinants produced by the
normalizing flow transformations.

Example:
    >>> from anomalib.models.image.fastflow.loss import FastflowLoss
    >>> criterion = FastflowLoss()
    >>> hidden_vars = [torch.randn(2, 64, 32, 32)]  # from NF blocks
    >>> jacobians = [torch.randn(2)]  # log det jacobians
    >>> loss = criterion(hidden_vars, jacobians)
    >>> loss.shape
    torch.Size([])

See Also:
    :class:`anomalib.models.image.fastflow.torch_model.FastflowModel`:
        PyTorch implementation of the FastFlow model architecture.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn


class FastflowLoss(nn.Module):
    """FastFlow Loss Module.

    Computes the negative log-likelihood loss used to train the FastFlow model. The loss
    combines the log-likelihood of the hidden variables with the log determinant of the
    Jacobian matrices from the normalizing flow transformations.
    """

    @staticmethod
    def forward(hidden_variables: list[torch.Tensor], jacobians: list[torch.Tensor]) -> torch.Tensor:
        """Calculate the FastFlow loss.

        The loss is computed as the negative log-likelihood of the hidden variables
        transformed by the normalizing flows, taking into account the Jacobian
        determinants of the transformations.

        Args:
            hidden_variables (list[torch.Tensor]): List of hidden variable tensors
                produced by the normalizing flow transformations. Each tensor has
                shape ``(N, C, H, W)`` where ``N`` is batch size.
            jacobians (list[torch.Tensor]): List of log determinants of Jacobian
                matrices for each normalizing flow transformation. Each tensor has
                shape ``(N,)`` where ``N`` is batch size.

        Returns:
            torch.Tensor: Scalar loss value combining the negative log-likelihood
                of hidden variables and Jacobian determinants.

        Example:
            >>> criterion = FastflowLoss()
            >>> h_vars = [torch.randn(2, 64, 32, 32)]  # hidden variables
            >>> jacs = [torch.randn(2)]  # log det jacobians
            >>> loss = criterion(h_vars, jacs)
        """
        loss = torch.tensor(0.0, device=hidden_variables[0].device)  # pylint: disable=not-callable
        for hidden_variable, jacobian in zip(hidden_variables, jacobians, strict=True):
            loss += torch.mean(0.5 * torch.sum(hidden_variable**2, dim=(1, 2, 3)) - jacobian)
        return loss
