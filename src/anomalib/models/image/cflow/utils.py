"""Helper functions for CFlow implementation.

This module provides utility functions used by the CFlow model implementation,
including:

- Log likelihood estimation
- 2D positional encoding generation
- Subnet and decoder network creation
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import math

import numpy as np
import torch
from FrEIA.framework import SequenceINN
from torch import nn

from anomalib.models.components.flow import AllInOneBlock

logger = logging.getLogger(__name__)


def get_logp(dim_feature_vector: int, p_u: torch.Tensor, logdet_j: torch.Tensor) -> torch.Tensor:
    """Calculate the log likelihood estimation.

    Args:
        dim_feature_vector (int): Dimension of the feature vector
        p_u (torch.Tensor): Random variable ``u`` sampled from the base distribution
        logdet_j (torch.Tensor): Log determinant of the Jacobian returned from the
            invertible decoder

    Returns:
        torch.Tensor: Log probability estimation

    Example:
        >>> dim = 128
        >>> p_u = torch.randn(32, dim)
        >>> logdet_j = torch.zeros(32)
        >>> logp = get_logp(dim, p_u, logdet_j)
    """
    ln_sqrt_2pi = -np.log(np.sqrt(2 * np.pi))  # ln(sqrt(2*pi))
    return dim_feature_vector * ln_sqrt_2pi - 0.5 * torch.sum(p_u**2, 1) + logdet_j


def positional_encoding_2d(condition_vector: int, height: int, width: int) -> torch.Tensor:
    """Create 2D positional encoding using sine and cosine functions.

    Creates an embedding to store relative position of feature vectors using
    sinusoidal functions at different frequencies.

    Args:
        condition_vector (int): Length of the condition vector (must be multiple
            of 4)
        height (int): Height of the positions grid
        width (int): Width of the positions grid

    Raises:
        ValueError: If ``condition_vector`` is not a multiple of 4

    Returns:
        torch.Tensor: Position encoding of shape
            ``(condition_vector, height, width)``

    Example:
        >>> encoding = positional_encoding_2d(128, 32, 32)
        >>> encoding.shape
        torch.Size([128, 32, 32])
    """
    if condition_vector % 4 != 0:
        msg = f"Cannot use sin/cos positional encoding with odd dimension (got dim={condition_vector})"
        raise ValueError(msg)
    pos_encoding = torch.zeros(condition_vector, height, width)
    # Each dimension use half of condition_vector
    condition_vector = condition_vector // 2
    div_term = torch.exp(torch.arange(0.0, condition_vector, 2) * -(math.log(1e4) / condition_vector))
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pos_encoding[0:condition_vector:2, :, :] = (
        torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pos_encoding[1:condition_vector:2, :, :] = (
        torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pos_encoding[condition_vector::2, :, :] = (
        torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    pos_encoding[condition_vector + 1 :: 2, :, :] = (
        torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    return pos_encoding


def subnet_fc(dims_in: int, dims_out: int) -> nn.Sequential:
    """Create a feed-forward subnetwork that predicts affine coefficients.

    Args:
        dims_in (int): Input dimensions
        dims_out (int): Output dimensions

    Returns:
        nn.Sequential: Feed-forward subnetwork with ReLU activation

    Example:
        >>> net = subnet_fc(64, 128)
        >>> x = torch.randn(32, 64)
        >>> out = net(x)
        >>> out.shape
        torch.Size([32, 128])
    """
    return nn.Sequential(nn.Linear(dims_in, 2 * dims_in), nn.ReLU(), nn.Linear(2 * dims_in, dims_out))


def cflow_head(
    condition_vector: int,
    coupling_blocks: int,
    clamp_alpha: float,
    n_features: int,
    permute_soft: bool = False,
) -> SequenceINN:
    """Create an invertible decoder network for CFlow.

    Args:
        condition_vector (int): Length of the condition vector
        coupling_blocks (int): Number of coupling blocks in the decoder
        clamp_alpha (float): Clamping value to avoid exploding values
        n_features (int): Number of decoder features
        permute_soft (bool, optional): Whether to sample the permutation matrix
            from SO(N) (True) or use hard permutations (False). Note that
            ``permute_soft=True`` is very slow for >512 dimensions.
            Defaults to False.

    Returns:
        SequenceINN: Invertible decoder network

    Example:
        >>> decoder = cflow_head(
        ...     condition_vector=128,
        ...     coupling_blocks=4,
        ...     clamp_alpha=1.9,
        ...     n_features=256
        ... )
    """
    coder = SequenceINN(n_features)
    logger.info("CNF coder: %d", n_features)
    for _ in range(coupling_blocks):
        coder.append(
            AllInOneBlock,
            cond=0,
            cond_shape=(condition_vector,),
            subnet_constructor=subnet_fc,
            affine_clamping=clamp_alpha,
            global_affine_type="SOFTPLUS",
            permute_soft=permute_soft,
        )
    return coder
