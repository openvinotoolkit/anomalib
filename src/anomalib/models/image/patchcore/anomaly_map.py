"""Anomaly Map Generator for the PatchCore model implementation.

This module generates anomaly heatmaps for the PatchCore model by upsampling
patch-level anomaly scores and applying Gaussian smoothing.

The anomaly map generation process involves:
1. Taking patch-level anomaly scores as input
2. Optionally upsampling scores to match input image dimensions
3. Applying Gaussian blur to smooth the final anomaly map

Example:
    >>> from anomalib.models.image.patchcore.anomaly_map import AnomalyMapGenerator
    >>> generator = AnomalyMapGenerator(sigma=4)
    >>> patch_scores = torch.randn(32, 1, 28, 28)  # (B, 1, H, W)
    >>> anomaly_map = generator(
    ...     patch_scores=patch_scores,
    ...     image_size=(224, 224)
    ... )

See Also:
    - :class:`anomalib.models.image.patchcore.lightning_model.Patchcore`:
        Lightning implementation of the PatchCore model
    - :class:`anomalib.models.components.GaussianBlur2d`:
        Gaussian blur module used for smoothing anomaly maps
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.models.components import GaussianBlur2d


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap.

    This class implements anomaly map generation for the PatchCore model by
    upsampling patch scores and applying Gaussian smoothing.

    Args:
        sigma (int, optional): Standard deviation for Gaussian smoothing kernel.
            Higher values produce smoother anomaly maps. Defaults to ``4``.

    Example:
        >>> generator = AnomalyMapGenerator(sigma=4)
        >>> patch_scores = torch.randn(32, 1, 28, 28)
        >>> anomaly_map = generator(patch_scores)
    """

    def __init__(
        self,
        sigma: int = 4,
    ) -> None:
        super().__init__()
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = GaussianBlur2d(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma), channels=1)

    def compute_anomaly_map(
        self,
        patch_scores: torch.Tensor,
        image_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        """Compute pixel-level anomaly heatmap from patch scores.

        Args:
            patch_scores (torch.Tensor): Patch-level anomaly scores with shape
                ``(B, 1, H, W)``
            image_size (tuple[int, int] | torch.Size | None, optional): Target
                size ``(H, W)`` to upsample anomaly map. If ``None``, keeps
                original size. Defaults to ``None``.

        Returns:
            torch.Tensor: Pixel-level anomaly scores after upsampling and
                smoothing, with shape ``(B, 1, H, W)``
        """
        if image_size is None:
            anomaly_map = patch_scores
        else:
            anomaly_map = F.interpolate(patch_scores, size=(image_size[0], image_size[1]))
        return self.blur(anomaly_map)

    def forward(
        self,
        patch_scores: torch.Tensor,
        image_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        """Generate smoothed anomaly map from patch scores.

        Args:
            patch_scores (torch.Tensor): Patch-level anomaly scores with shape
                ``(B, 1, H, W)``
            image_size (tuple[int, int] | torch.Size | None, optional): Target
                size ``(H, W)`` to upsample anomaly map. If ``None``, keeps
                original size. Defaults to ``None``.

        Example:
            >>> generator = AnomalyMapGenerator(sigma=4)
            >>> patch_scores = torch.randn(32, 1, 28, 28)
            >>> anomaly_map = generator(
            ...     patch_scores=patch_scores,
            ...     image_size=(224, 224)
            ... )

        Returns:
            torch.Tensor: Anomaly heatmap after upsampling and smoothing,
                with shape ``(B, 1, H, W)``
        """
        return self.compute_anomaly_map(patch_scores, image_size)
