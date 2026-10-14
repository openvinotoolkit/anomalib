# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Anomaly-map generation for MH-PatchCore."""

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.models.components import GaussianBlur2d

_BLUR_SIGMA = 4.0
_BLUR_PADDING = int(4.0 * _BLUR_SIGMA + 0.5)


class AnomalyMapGenerator(nn.Module):
    """Resize and smooth MH-PatchCore patch scores."""

    def __init__(self) -> None:
        super().__init__()
        self.blur = GaussianBlur2d(sigma=_BLUR_SIGMA, channels=1, padding="valid")

    @staticmethod
    def _symmetric_pad(input_tensor: torch.Tensor) -> torch.Tensor:
        height, width = input_tensor.shape[-2:]
        if height <= _BLUR_PADDING or width <= _BLUR_PADDING:
            msg = f"Anomaly-map dimensions must be greater than {_BLUR_PADDING} for Gaussian smoothing."
            raise ValueError(msg)
        horizontal = torch.cat(
            (
                input_tensor[..., :_BLUR_PADDING].flip(-1),
                input_tensor,
                input_tensor[..., -_BLUR_PADDING:].flip(-1),
            ),
            dim=-1,
        )
        return torch.cat(
            (
                horizontal[..., :_BLUR_PADDING, :].flip(-2),
                horizontal,
                horizontal[..., -_BLUR_PADDING:, :].flip(-2),
            ),
            dim=-2,
        )

    @staticmethod
    def resize(
        patch_scores: torch.Tensor,
        image_size: tuple[int, int] | torch.Size,
    ) -> torch.Tensor:
        """Resize patch scores to the input image size.

        Args:
            patch_scores (torch.Tensor): Patch scores with shape ``[B, 1, H, W]``.
            image_size (tuple[int, int] | torch.Size): Target height and width.

        Returns:
            torch.Tensor: Bilinearly resized scores with shape ``[B, 1, H_out, W_out]``.
        """
        return F.interpolate(patch_scores, size=image_size, mode="bilinear", align_corners=False)

    def forward(
        self,
        patch_scores: torch.Tensor,
        image_size: tuple[int, int] | torch.Size,
    ) -> torch.Tensor:
        """Generate a smoothed full-resolution anomaly map.

        Args:
            patch_scores (torch.Tensor): Patch scores with shape ``[B, 1, H, W]``.
            image_size (tuple[int, int] | torch.Size): Target height and width.

        Returns:
            torch.Tensor: Anomaly maps with shape ``[B, 1, H_out, W_out]``.

        Raises:
            ValueError: If either output dimension is too small for the fixed
                Gaussian kernel.
        """
        resized_scores = self.resize(patch_scores, image_size)
        return self.blur(self._symmetric_pad(resized_scores))
