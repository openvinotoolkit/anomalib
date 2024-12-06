"""Anomaly generator for the DSR model implementation."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor, nn
from torchvision.transforms import v2

from anomalib.data.utils.generators.perlin import generate_perlin_noise


class DsrAnomalyGenerator(nn.Module):
    """Anomaly generator of the DSR model.

    The anomaly is generated using a Perlin noise generator on the two quantized representations of an image.
    This generator is only used during the second phase of training! The third phase requires generating
    smudges over the input images.

    Args:
        p_anomalous (float, optional): Probability to generate an anomalous image.
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
        """Generate an anomalous mask.

        Args:
            height (int): Height of generated mask.
            width (int): Width of generated mask.

        Returns:
            Tensor: Generated mask.
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
        """Generate anomalous augmentations for a batch of input images.

        Args:
            batch (Tensor): Batch of input images

        Returns:
            Tensor: Ground truth masks corresponding to the anomalous perturbations.
        """
        batch_size, _, height, width = batch.shape

        # Collect perturbations
        masks_list: list[Tensor] = []
        for _ in range(batch_size):
            if torch.rand(1) > self.p_anomalous:  # include normal samples
                masks_list.append(torch.zeros((1, height, width)))
            else:
                mask = self.generate_anomaly(height, width)
                masks_list.append(mask)

        return torch.stack(masks_list).to(batch.device)
