"""Anomaly generator for the DSR model implementation.

This module implements an anomaly generator that creates synthetic anomalies
using Perlin noise. The generator is used during the second phase of DSR model
training to create anomalous samples.

Example:
    >>> from anomalib.models.image.dsr.anomaly_generator import DsrAnomalyGenerator
    >>> generator = DsrAnomalyGenerator(p_anomalous=0.5)
    >>> batch = torch.randn(8, 3, 256, 256)
    >>> masks = generator.augment_batch(batch)
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor, nn
from torchvision.transforms import v2

from anomalib.data.utils.generators.perlin import generate_perlin_noise


class DsrAnomalyGenerator(nn.Module):
    """Anomaly generator for the DSR model.

    The generator creates synthetic anomalies by applying Perlin noise to images.
    It is used during the second phase of DSR model training. The third phase
    uses a different approach with smudge-based anomalies.

    Args:
        p_anomalous (float, optional): Probability of generating an anomalous
            image. Defaults to ``0.5``.

    Example:
        >>> generator = DsrAnomalyGenerator(p_anomalous=0.7)
        >>> batch = torch.randn(4, 3, 256, 256)
        >>> masks = generator.augment_batch(batch)
        >>> assert masks.shape == (4, 1, 256, 256)
    """

    def __init__(
        self,
        p_anomalous: float = 0.5,
    ) -> None:
        super().__init__()

        self.p_anomalous = p_anomalous
        # Replace imgaug with torchvision transform
        self.rot = v2.RandomAffine(degrees=(-90, 90))

    def generate_anomaly(self, height: int, width: int) -> Tensor:
        """Generate an anomalous mask using Perlin noise.

        Args:
            height (int): Height of the mask to generate.
            width (int): Width of the mask to generate.

        Returns:
            Tensor: Binary mask of shape ``(1, height, width)`` where ``1``
                indicates anomalous regions.

        Example:
            >>> generator = DsrAnomalyGenerator()
            >>> mask = generator.generate_anomaly(256, 256)
            >>> assert mask.shape == (1, 256, 256)
            >>> assert torch.all((mask >= 0) & (mask <= 1))
        """
        min_perlin_scale = 0
        perlin_scale = 6
        perlin_scalex = int(2 ** torch.randint(min_perlin_scale, perlin_scale, (1,)).item())
        perlin_scaley = int(2 ** torch.randint(min_perlin_scale, perlin_scale, (1,)).item())
        threshold = 0.5

        # Generate perlin noise using the new function
        perlin_noise = generate_perlin_noise(height, width, scale=(perlin_scalex, perlin_scaley))

        # Apply random rotation
        perlin_noise = perlin_noise.unsqueeze(0)  # Add channel dimension for transform
        perlin_noise = self.rot(perlin_noise).squeeze(0)  # Remove channel dimension

        # Create binary mask
        mask = (perlin_noise > threshold).float()
        return mask.unsqueeze(0)  # Add channel dimension [1, H, W]

    def augment_batch(self, batch: Tensor) -> Tensor:
        """Generate anomalous masks for a batch of images.

        Args:
            batch (Tensor): Input batch of images of shape
                ``(batch_size, channels, height, width)``.

        Returns:
            Tensor: Batch of binary masks of shape
                ``(batch_size, 1, height, width)`` where ``1`` indicates
                anomalous regions.

        Example:
            >>> generator = DsrAnomalyGenerator()
            >>> batch = torch.randn(8, 3, 256, 256)
            >>> masks = generator.augment_batch(batch)
            >>> assert masks.shape == (8, 1, 256, 256)
            >>> assert torch.all((masks >= 0) & (masks <= 1))
        """
        batch_size, _, height, width = batch.shape

        # Collect perturbations
        masks_list: list[Tensor] = []
        for _ in range(batch_size):
            if torch.rand(1) > self.p_anomalous:  # include normal samples
                masks_list.append(torch.zeros((1, height, width), device=batch.device))
            else:
                mask = self.generate_anomaly(height, width)
                masks_list.append(mask)

        return torch.stack(masks_list).to(batch.device)
