"""Gaussian blurring implementation using PyTorch.

This module provides a 2D Gaussian blur filter implementation that pre-computes
the Gaussian kernel during initialization for efficiency.

Example:
    >>> import torch
    >>> from anomalib.models.components.filters import GaussianBlur2d
    >>> # Create a Gaussian blur filter
    >>> blur = GaussianBlur2d(sigma=1.0, channels=3)
    >>> # Apply blur to input tensor
    >>> input_tensor = torch.randn(1, 3, 256, 256)
    >>> blurred = blur(input_tensor)
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from kornia.filters import get_gaussian_kernel2d
from kornia.filters.filter import _compute_padding
from kornia.filters.kernels import normalize_kernel2d
from torch import nn
from torch.nn import functional as F  # noqa: N812


def compute_kernel_size(sigma_val: float) -> int:
    """Compute kernel size from sigma value.

    The kernel size is calculated as 2 * (4 * sigma + 0.5) + 1 to ensure it
    captures the significant part of the Gaussian distribution.

    Args:
        sigma_val (float): Standard deviation value for the Gaussian kernel.

    Returns:
        int: Computed kernel size (always odd).

    Example:
        >>> compute_kernel_size(1.0)
        9
        >>> compute_kernel_size(2.0)
        17
    """
    return 2 * int(4.0 * sigma_val + 0.5) + 1


class GaussianBlur2d(nn.Module):
    """2D Gaussian blur filter with pre-computed kernel.

    Unlike some implementations, this class pre-computes the Gaussian kernel
    during initialization rather than computing it during the forward pass.
    This approach is more efficient but requires specifying the number of
    input channels upfront.

    Args:
        sigma (float | tuple[float, float]): Standard deviation(s) for the
            Gaussian kernel. If a single float is provided, it's used for both
            dimensions.
        channels (int): Number of input channels. Defaults to 1.
        kernel_size (int | tuple[int, int] | None): Size of the Gaussian
            kernel. If ``None``, computed from sigma. Defaults to ``None``.
        normalize (bool): Whether to normalize the kernel so its elements sum
            to 1. Defaults to ``True``.
        border_type (str): Padding mode for border handling. Options are
            'reflect', 'replicate', etc. Defaults to "reflect".
        padding (str): Padding strategy. Either 'same' or 'valid'.
            Defaults to "same".

    Example:
        >>> import torch
        >>> blur = GaussianBlur2d(sigma=1.0, channels=3)
        >>> x = torch.randn(1, 3, 64, 64)
        >>> output = blur(x)
        >>> output.shape
        torch.Size([1, 3, 64, 64])
    """

    def __init__(
        self,
        sigma: float | tuple[float, float],
        channels: int = 1,
        kernel_size: int | tuple[int, int] | None = None,
        normalize: bool = True,
        border_type: str = "reflect",
        padding: str = "same",
    ) -> None:
        super().__init__()
        sigma = sigma if isinstance(sigma, tuple) else (sigma, sigma)
        self.channels = channels

        if kernel_size is None:
            kernel_size = (compute_kernel_size(sigma[0]), compute_kernel_size(sigma[1]))
        else:
            kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        self.kernel: torch.Tensor
        self.register_buffer("kernel", get_gaussian_kernel2d(kernel_size=kernel_size, sigma=sigma))
        if normalize:
            self.kernel = normalize_kernel2d(self.kernel)

        self.kernel = self.kernel.view(1, 1, *self.kernel.shape[-2:])

        self.kernel = self.kernel.expand(self.channels, -1, -1, -1)
        self.border_type = border_type
        self.padding = padding
        self.height, self.width = self.kernel.shape[-2:]
        self.padding_shape = _compute_padding([self.height, self.width])

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur to input tensor.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape
                ``(B, C, H, W)``.

        Returns:
            torch.Tensor: Blurred output tensor. If padding is 'same',
                output shape matches input. If 'valid', output is smaller.

        Example:
            >>> blur = GaussianBlur2d(sigma=1.0, channels=1)
            >>> x = torch.ones(1, 1, 5, 5)
            >>> output = blur(x)
            >>> output.shape
            torch.Size([1, 1, 5, 5])
        """
        batch, channel, height, width = input_tensor.size()

        if self.padding == "same":
            input_tensor = F.pad(input_tensor, self.padding_shape, mode=self.border_type)

        # convolve the tensor with the kernel.
        output = F.conv2d(input_tensor, self.kernel, groups=self.channels, padding=0, stride=1)

        if self.padding == "same":
            out = output.view(batch, channel, height, width)
        else:
            out = output.view(batch, channel, height - self.height + 1, width - self.width + 1)

        return out
