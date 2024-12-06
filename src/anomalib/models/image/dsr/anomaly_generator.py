"""Anomaly generator for the DSR model implementation."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import imgaug.augmenters as iaa
import numpy as np
import torch
from torch import Tensor, nn

from anomalib.data.utils.generators.perlin import _rand_perlin_2d_np


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
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

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
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        threshold = 0.5
        perlin_noise_np = _rand_perlin_2d_np((height, width), (perlin_scalex, perlin_scaley))
        perlin_noise_np = self.rot(image=perlin_noise_np)
        mask = np.where(perlin_noise_np > threshold, np.ones_like(perlin_noise_np), np.zeros_like(perlin_noise_np))
        mask = np.expand_dims(mask, axis=2).astype(np.float32)

        return torch.from_numpy(mask)

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
                masks_list.append(mask.permute((2, 0, 1)))

        return torch.stack(masks_list).to(batch.device)
