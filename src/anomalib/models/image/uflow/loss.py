"""Loss function implementation for the U-Flow model.

This module implements the loss function used to train the U-Flow model for anomaly
detection as described in  <https://arxiv.org/pdf/2211.12353.pdf>`_.
The loss combines:

- A likelihood term based on the hidden variables
- A Jacobian determinant term from the normalizing flow

Example:
    >>> from anomalib.models.image.uflow.loss import UFlowLoss
    >>> loss_fn = UFlowLoss()
    >>> hidden_vars = [torch.randn(2, 64, 32, 32)]
    >>> jacobians = [torch.randn(2)]
    >>> loss = loss_fn(hidden_vars, jacobians)

See Also:
    - :class:`UFlowLoss`: Main loss function implementation
    - :class:`UflowModel`: PyTorch model using this loss
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor, nn


class UFlowLoss(nn.Module):
    """Loss function for training the U-Flow model.

    This class implements the loss function used to train the U-Flow model.
    The loss combines:

    1. A likelihood term based on the hidden variables (``lpz``)
    2. A Jacobian determinant term from the normalizing flow

    The total loss is computed as:
    ``loss = mean(lpz - jacobians)``

    Example:
        >>> from anomalib.models.image.uflow.loss import UFlowLoss
        >>> loss_fn = UFlowLoss()
        >>> hidden_vars = [torch.randn(2, 64, 32, 32)]  # List of hidden variables
        >>> jacobians = [torch.randn(2)]  # List of log Jacobian determinants
        >>> loss = loss_fn(hidden_vars, jacobians)
        >>> loss.shape
        torch.Size([])

    See Also:
        - :class:`UflowModel`: PyTorch model using this loss function
        - :class:`Uflow`: Lightning implementation using this loss
    """

    @staticmethod
    def forward(hidden_variables: list[Tensor], jacobians: list[Tensor]) -> Tensor:
        """Calculate the UFlow loss.

        Args:
            hidden_variables (list[Tensor]): List of hidden variable tensors from the
                normalizing flow transformation f: X -> Z. Each tensor has shape
                ``(batch_size, channels, height, width)``.
            jacobians (list[Tensor]): List of log Jacobian determinant tensors from the
                flow transformation. Each tensor has shape ``(batch_size,)``.

        Returns:
            Tensor: Scalar loss value combining the likelihood of hidden variables and
                the log Jacobian determinants.
        """
        lpz = torch.sum(torch.stack([0.5 * torch.sum(z_i**2, dim=(1, 2, 3)) for z_i in hidden_variables], dim=0))
        return torch.mean(lpz - jacobians)
