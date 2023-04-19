"""Helper functions for CFlow implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math

import numpy as np
import torch
from FrEIA.framework import SequenceINN
from torch import Tensor, nn

from anomalib.models.components.flow import AllInOneBlock

logger = logging.getLogger(__name__)


def get_logp(dim_feature_vector: int, p_u: Tensor, logdet_j: Tensor) -> Tensor:
    """Returns the log likelihood estimation.

    Args:
        dim_feature_vector (int): Dimensions of the condition vector
        p_u (Tensor): Random variable u
        logdet_j (Tensor): log of determinant of jacobian returned from the invertable decoder

    Returns:
        Tensor: Log probability
    """
    ln_sqrt_2pi = -np.log(np.sqrt(2 * np.pi))  # ln(sqrt(2*pi))
    logp = dim_feature_vector * ln_sqrt_2pi - 0.5 * torch.sum(p_u**2, 1) + logdet_j
    return logp


def positional_encoding_2d(condition_vector: int, height: int, width: int) -> Tensor:
    """Creates embedding to store relative position of the feature vector using sine and cosine functions.

    Args:
        condition_vector (int): Length of the condition vector
        height (int): H of the positions
        width (int): W of the positions

    Raises:
        ValueError: Cannot generate encoding with conditional vector length not as multiple of 4

    Returns:
        Tensor: condition_vector x HEIGHT x WIDTH position matrix
    """
    if condition_vector % 4 != 0:
        raise ValueError(f"Cannot use sin/cos positional encoding with odd dimension (got dim={condition_vector})")
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
    """Subnetwork which predicts the affine coefficients.

    Args:
        dims_in (int): input dimensions
        dims_out (int): output dimensions

    Returns:
        nn.Sequential: Feed-forward subnetwork
    """
    return nn.Sequential(nn.Linear(dims_in, 2 * dims_in), nn.ReLU(), nn.Linear(2 * dims_in, dims_out))


def cflow_head(
    condition_vector: int, coupling_blocks: int, clamp_alpha: float, n_features: int, permute_soft: bool = False
) -> SequenceINN:
    """Create invertible decoder network.

    Args:
        condition_vector (int): length of the condition vector
        coupling_blocks (int): number of coupling blocks to build the decoder
        clamp_alpha (float): clamping value to avoid exploding values
        n_features (int): number of decoder features
        permute_soft (bool): Whether to sample the permutation matrix :math:`R` from :math:`SO(N)`,
            or to use hard permutations instead. Note, ``permute_soft=True`` is very slow
            when working with >512 dimensions.

    Returns:
        SequenceINN: decoder network block
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
