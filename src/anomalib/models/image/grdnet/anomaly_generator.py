# Original Code
# Copyright (c) 2024-2026 Niccolò Ferrari
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Synthetic anomaly generation for GRD-Net."""

from typing import Literal

import torch
from torch import nn

from anomalib.data.utils.generators import generate_perlin_noise

TextureSource = Literal["random", "image"]


class GRDNetAnomalyGenerator(nn.Module):
    """Generate paper-canonical synthetic anomalies for GRD-Net.

    Args:
        texture_source: Source used to fill the synthetic anomaly. ``"random"``
            samples independent RGB values and ``"image"`` circularly shifts
            each input tile.
        probability: Per-tile probability of applying an anomaly.
        min_mask_area: Minimum number of active pixels in a sampled Perlin mask.
        max_mask_attempts: Maximum Perlin samples used to satisfy ``min_mask_area``.
    """

    def __init__(
        self,
        texture_source: TextureSource = "random",
        probability: float = 0.75,
        min_mask_area: int = 25,
        max_mask_attempts: int = 8,
    ) -> None:
        super().__init__()
        if texture_source not in {"random", "image"}:
            msg = f"Unknown texture source {texture_source!r}. Expected 'random' or 'image'."
            raise ValueError(msg)
        if isinstance(probability, bool) or not 0.0 <= probability <= 1.0:
            msg = f"Probability must be in [0, 1], got {probability!r}."
            raise ValueError(msg)
        if isinstance(min_mask_area, bool) or min_mask_area < 1:
            msg = f"Minimum mask area must be a positive integer, got {min_mask_area!r}."
            raise ValueError(msg)
        if isinstance(max_mask_attempts, bool) or max_mask_attempts < 1:
            msg = f"Maximum mask attempts must be a positive integer, got {max_mask_attempts!r}."
            raise ValueError(msg)

        self.texture_source = texture_source
        self.probability = probability
        self.min_mask_area = min_mask_area
        self.max_mask_attempts = max_mask_attempts

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply synthetic anomalies to an image-tile batch.

        Args:
            images: Float image tiles with shape ``[N, 3, H, W]`` and values in
                ``[0, 1]``.

        Returns:
            Tuple containing perturbed images, sampled textures, binary anomaly
            masks, and per-tile blend factors.

        Raises:
            ValueError: If the input shape, range, or configured mask area is invalid.
            RuntimeError: If a valid Perlin mask cannot be sampled within the attempt limit.
        """
        _validate_images(images)
        height, width = images.shape[-2:]
        if self.min_mask_area > height * width:
            msg = f"Minimum mask area {self.min_mask_area} exceeds tile area {height * width}."
            raise ValueError(msg)

        textures = torch.zeros_like(images)
        masks = torch.zeros(
            (images.shape[0], 1, height, width),
            device=images.device,
            dtype=images.dtype,
        )
        for index, image in enumerate(images):
            if torch.rand((), device=images.device).item() >= self.probability:
                continue
            masks[index] = self._sample_mask(height, width, images.device, images.dtype)
            textures[index] = _sample_texture(image, self.texture_source)

        beta = torch.empty(
            (images.shape[0], 1, 1, 1),
            device=images.device,
            dtype=images.dtype,
        ).uniform_(0.5, 1.0)
        perturbed = _blend_anomaly(images, textures, masks, beta)
        return perturbed, textures, masks, beta

    def _sample_mask(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sample a Perlin mask that satisfies the configured area."""
        for _ in range(self.max_mask_attempts):
            scale_x = 2 ** int(torch.randint(0, 6, (), device=device).item())
            scale_y = 2 ** int(torch.randint(0, 6, (), device=device).item())
            perlin = generate_perlin_noise(height, width, scale=(scale_x, scale_y), device=device)
            mask = (perlin > 0.5).to(dtype=dtype).unsqueeze(0)
            if int(mask.sum().item()) >= self.min_mask_area:
                return mask
        msg = (
            f"Unable to sample a Perlin mask with at least {self.min_mask_area} active pixels "
            f"after {self.max_mask_attempts} attempts."
        )
        raise RuntimeError(msg)


def _sample_texture(image: torch.Tensor, source: TextureSource) -> torch.Tensor:
    """Sample one RGB texture from the configured source."""
    if source == "random":
        return torch.rand_like(image)

    height, width = image.shape[-2:]
    shift_y = int(torch.randint(0, height, (), device=image.device).item())
    shift_x = int(torch.randint(0, width, (), device=image.device).item())
    if shift_y == 0 and shift_x == 0:
        shift_x = 1 if width > 1 else 0
        shift_y = 1 if width == 1 and height > 1 else shift_y
    if shift_y == 0 and shift_x == 0:
        msg = "Image textures require at least one spatial dimension larger than one."
        raise ValueError(msg)
    return torch.roll(image, shifts=(shift_y, shift_x), dims=(-2, -1))


def _blend_anomaly(
    images: torch.Tensor,
    textures: torch.Tensor,
    masks: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Alpha-blend textures only inside binary anomaly masks."""
    return ((1.0 - masks) * images + masks * ((1.0 - beta) * images + beta * textures)).clamp(0.0, 1.0)


def _validate_images(images: torch.Tensor) -> None:
    """Validate an RGB float image-tile batch."""
    if images.ndim != 4 or images.shape[1] != 3:
        msg = f"Expected RGB image tiles with shape [N, 3, H, W], got {tuple(images.shape)}."
        raise ValueError(msg)
    if not images.is_floating_point():
        msg = f"Expected floating-point image tiles, got {images.dtype}."
        raise ValueError(msg)
    if not torch.isfinite(images).all() or torch.any((images < 0.0) | (images > 1.0)):
        msg = "Image tiles must contain finite values in [0, 1]."
        raise ValueError(msg)
