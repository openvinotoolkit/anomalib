"""Helper functions for generating Perlin noise."""

# Original Code
# Copyright (c) 2021 VitjanZ
# https://github.com/VitjanZ/DRAEM.
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=invalid-name

import math
from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor


def lerp_np(x, y, w):
    """Helper function."""
    fin_out = (y - x) * w + x
    return fin_out


def rand_perlin_2d_octaves_np(shape, res, octaves=1, persistence=0.5):
    """Generate Perlin noise parameterized by the octaves method. Numpy version."""
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise


def generate_perlin_noise_2d(shape, res):
    """Fractal perlin noise."""

    def f(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def random_2d_perlin(
    shape: Tuple,
    res: Tuple[Union[int, Tensor], Union[int, Tensor]],
    fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3,
) -> Union[np.ndarray, Tensor]:
    """Returns a random 2d perlin noise array.

    Args:
        shape (Tuple): Shape of the 2d map.
        res (Tuple[Union[int, Tensor]]): Tuple of scales for perlin noise for height and width dimension.
        fade (_type_, optional): Function used for fading the resulting 2d map.
            Defaults to equation 6*t**5-15*t**4+10*t**3.

    Returns:
        Union[np.ndarray, Tensor]: Random 2d-array/tensor generated using perlin noise.
    """
    if isinstance(res[0], int):
        result = _rand_perlin_2d_np(shape, res, fade)
    elif isinstance(res[0], Tensor):
        result = _rand_perlin_2d(shape, res, fade)
    else:
        raise TypeError(f"got scales of type {type(res[0])}")
    return result


def _rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3):
    """Generate a random image containing Perlin noise. Numpy version."""
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    def tile_grads(slice1, slice2):
        return np.repeat(np.repeat(gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]], d[0], axis=0), d[1], axis=1)

    def dot(grad, shift):
        return (
            np.stack((grid[: shape[0], : shape[1], 0] + shift[0], grid[: shape[0], : shape[1], 1] + shift[1]), axis=-1)
            * grad[: shape[0], : shape[1]]
        ).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[: shape[0], : shape[1]])
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])


def _rand_perlin_2d(shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3):
    """Generate a random image containing Perlin noise. PyTorch version."""
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim=-1) % 1
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    def tile_grads(slice1, slice2):
        return (
            gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]]
            .repeat_interleave(d[0], 0)
            .repeat_interleave(d[1], 1)
        )

    def dot(grad, shift):
        return (
            torch.stack(
                (grid[: shape[0], : shape[1], 0] + shift[0], grid[: shape[0], : shape[1], 1] + shift[1]), dim=-1
            )
            * grad[: shape[0], : shape[1]]
        ).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])

    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[: shape[0], : shape[1]])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])


def rand_perlin_2d_octaves(shape, res, octaves=1, persistence=0.5):
    """Generate Perlin noise parameterized by the octaves method. PyTorch version."""
    noise = torch.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * _rand_perlin_2d(shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise
