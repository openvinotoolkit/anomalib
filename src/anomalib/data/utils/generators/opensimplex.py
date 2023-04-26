"""Helper functions for generating Open Simplex noise."""

### add license
from __future__ import annotations

from enum import Enum
from math import floor

import numpy as np
import torch
from torch import Tensor
try:
    from numba import njit, prange
except ImportError:
    prange = range

    def njit(*args, **kwargs):
        def wrapper(func):
            return func

        return wrapper


class SimplexConstant(Enum):
    """Constant for simplex noise"""

    GRADIENTS2 = np.array([
         5,  2,  2,  5,
        -5,  2, -2,  5,
         5, -2,  2, -5,
        -5, -2, -2, -5,
    ], dtype=np.int64)
    STRETCH_CONSTANT2 = -0.211324865405187    # (1/Math.sqrt(2+1)-1)/2
    SQUISH_CONSTANT2 = 0.366025403784439      # (Math.sqrt(2+1)-1)/2
    NORM_CONSTANT2 = 47


@njit(cache=True)
def _extrapolate2(perm, xsb, ysb, dx, dy):
    index = perm[(perm[xsb & 0xFF] + ysb) & 0xFF] & 0x0E
    g1, g2 = SimplexConstant.GRADIENTS2[index : index + 2]
    return g1 * dx + g2 * dy


@njit(cache=True)
def _simplex_noise2d(x, y, perm):
    # Place input coordinates onto grid.
    stretch_offset = (x + y) * SimplexConstant.STRETCH_CONSTANT2
    xs = x + stretch_offset
    ys = y + stretch_offset

    # Floor to get grid coordinates of rhombus (stretched square) super-cell origin.
    xsb = floor(xs)
    ysb = floor(ys)

    # Skew out to get actual coordinates of rhombus origin. We'll need these later.
    squish_offset = (xsb + ysb) * SimplexConstant.SQUISH_CONSTANT2
    xb = xsb + squish_offset
    yb = ysb + squish_offset

    # Compute grid coordinates relative to rhombus origin.
    xins = xs - xsb
    yins = ys - ysb

    # Sum those together to get a value that determines which region we're in.
    in_sum = xins + yins

    # Positions relative to origin point.
    dx0 = x - xb
    dy0 = y - yb

    value = 0

    # Contribution (1,0)
    dx1 = dx0 - 1 - SimplexConstant.SQUISH_CONSTANT2
    dy1 = dy0 - 0 - SimplexConstant.SQUISH_CONSTANT2
    attn1 = 2 - dx1 * dx1 - dy1 * dy1
    if attn1 > 0:
        attn1 *= attn1
        value += attn1 * attn1 * _extrapolate2(perm, xsb + 1, ysb + 0, dx1, dy1)

    # Contribution (0,1)
    dx2 = dx0 - 0 - SimplexConstant.SQUISH_CONSTANT2
    dy2 = dy0 - 1 - SimplexConstant.SQUISH_CONSTANT2
    attn2 = 2 - dx2 * dx2 - dy2 * dy2
    if attn2 > 0:
        attn2 *= attn2
        value += attn2 * attn2 * _extrapolate2(perm, xsb + 0, ysb + 1, dx2, dy2)

    if in_sum <= 1:  # We're inside the triangle (2-Simplex) at (0,0)
        zins = 1 - in_sum
        if zins > xins or zins > yins:  # (0,0) is one of the closest two triangular vertices
            if xins > yins:
                xsv_ext = xsb + 1
                ysv_ext = ysb - 1
                dx_ext = dx0 - 1
                dy_ext = dy0 + 1
            else:
                xsv_ext = xsb - 1
                ysv_ext = ysb + 1
                dx_ext = dx0 + 1
                dy_ext = dy0 - 1
        else:  # (1,0) and (0,1) are the closest two vertices.
            xsv_ext = xsb + 1
            ysv_ext = ysb + 1
            dx_ext = dx0 - 1 - 2 * SimplexConstant.SQUISH_CONSTANT2
            dy_ext = dy0 - 1 - 2 * SimplexConstant.SQUISH_CONSTANT2
    else:  # We're inside the triangle (2-Simplex) at (1,1)
        zins = 2 - in_sum
        if zins < xins or zins < yins:  # (0,0) is one of the closest two triangular vertices
            if xins > yins:
                xsv_ext = xsb + 2
                ysv_ext = ysb + 0
                dx_ext = dx0 - 2 - 2 * SimplexConstant.SQUISH_CONSTANT2
                dy_ext = dy0 + 0 - 2 * SimplexConstant.SQUISH_CONSTANT2
            else:
                xsv_ext = xsb + 0
                ysv_ext = ysb + 2
                dx_ext = dx0 + 0 - 2 * SimplexConstant.SQUISH_CONSTANT2
                dy_ext = dy0 - 2 - 2 * SimplexConstant.SQUISH_CONSTANT2
        else:  # (1,0) and (0,1) are the closest two vertices.
            dx_ext = dx0
            dy_ext = dy0
            xsv_ext = xsb
            ysv_ext = ysb
        xsb += 1
        ysb += 1
        dx0 = dx0 - 1 - 2 * SimplexConstant.SQUISH_CONSTANT2
        dy0 = dy0 - 1 - 2 * SimplexConstant.SQUISH_CONSTANT2

    # Contribution (0,0) or (1,1)
    attn0 = 2 - dx0 * dx0 - dy0 * dy0
    if attn0 > 0:
        attn0 *= attn0
        value += attn0 * attn0 * _extrapolate2(perm, xsb, ysb, dx0, dy0)

    # Extra Vertex
    attn_ext = 2 - dx_ext * dx_ext - dy_ext * dy_ext
    if attn_ext > 0:
        attn_ext *= attn_ext
        value += attn_ext * attn_ext * _extrapolate2(perm, xsv_ext, ysv_ext, dx_ext, dy_ext)

    return value / SimplexConstant.NORM_CONSTANT2


def random_2d_simplex(
    shape: tuple[int | Tensor, int | Tensor],
)-> np.ndarray | Tensor:
    """Generate a random image containing Simplex noise.

     Args:
        shape (tuple): Shape of the 2d map.
        res (tuple[int | Tensor, int | Tensor]): Tuple of scales for simplex noise for height and width dimension.

    Returns:
        np.ndarray | Tensor: Random 2d-array/tensor generated using simplex noise.
    """

    if isinstance(shape[0], int):
        result = np.zeros(shape)
    elif isinstance(shape[0], Tensor):
        result = torch.zeros(shape)
    else:
        raise TypeError(f"got scales of type {type(shape[0])}")
    
    for y in range(shape[0]):
        for x in range(shape[1]):
            result[x, y] = _simplex_noise2d(x / shape[1], y / shape[0])
    result = (result + 1) * 0.5 # normalize to 0-1
    return result
