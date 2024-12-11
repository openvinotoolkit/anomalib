"""Anomaly generator for the SuperSimplenet model implementation."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from anomalib.data.utils.generators.perlin import _rand_perlin_2d


class SSNAnomalyGenerator(nn.Module):
    """Anomaly generator of the SuperSimpleNet model."""

    def __init__(
        self,
        noise_mean: float,
        noise_std: float,
        threshold: float,
        perlin_range: tuple[int, int] = (0, 6),
    ) -> None:
        super().__init__()

        self.noise_mean = noise_mean
        self.noise_std = noise_std

        self.threshold = threshold

        self.min_perlin_scale = perlin_range[0]
        self.max_perlin_scale = perlin_range[1]

    @staticmethod
    def next_power_2(num: int) -> int:
        """Get the next power of 2 for given number.

        Args:
            num (int): value of interest

        Returns:
            next power of 2 value for given number
        """
        return 1 << (num - 1).bit_length()

    def generate_perlin(self, batches: int, height: int, width: int) -> torch.Tensor:
        """Generate 2d perlin noise masks with dims [b, 1, h, w].

        Args:
            batches (int): number of batches (different masks)
            height (int): height of features
            width (int): width of features

        Returns:
            tensor with b perlin binarized masks
        """
        perlin = []
        for _ in range(batches):
            # get scale of perlin in x and y direction
            perlin_scalex = 2 ** (
                torch.randint(
                    self.min_perlin_scale,
                    self.max_perlin_scale,
                    (1,),
                ).item()
            )
            perlin_scaley = 2 ** (
                torch.randint(
                    self.min_perlin_scale,
                    self.max_perlin_scale,
                    (1,),
                ).item()
            )

            perlin_height = self.next_power_2(height)
            perlin_width = self.next_power_2(width)

            perlin_noise = _rand_perlin_2d(
                (perlin_height, perlin_width),
                (perlin_scalex, perlin_scaley),
            )
            # original is power of 2 scale, so fit to our size
            perlin_noise = F.interpolate(
                perlin_noise.reshape(1, 1, perlin_height, perlin_width),
                size=(height, width),
                mode="bilinear",
            )
            # binarize
            perlin_thr = torch.where(perlin_noise > self.threshold, 1, 0)

            # 50% of anomaly
            if torch.rand(1).item() > 0.5:
                perlin_thr = torch.zeros_like(perlin_thr)

            perlin.append(perlin_thr)
        return torch.cat(perlin)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate anomaly on features using thresholded perlin noise and Gaussian noise.

        Also update GT masks and labels with new anomaly information.

        Args:
            features: input features.
            mask: GT masks.
            labels: GT labels.

        Returns:
            perturbed features, updated GT masks and labels.
        """
        b, _, h, w = features.shape

        # duplicate
        features = torch.cat((features, features))
        mask = torch.cat((mask, mask))
        labels = torch.cat((labels, labels))

        noise = torch.normal(
            mean=self.noise_mean,
            std=self.noise_std,
            size=features.shape,
            device=features.device,
            requires_grad=False,
        )

        # mask indicating which regions will have noise applied
        # [B * 2, 1, H, W] initial all masked as anomalous
        noise_mask = torch.ones(
            b * 2,
            1,
            h,
            w,
            device=features.device,
            requires_grad=False,
        )

        # no overlap: don't apply to already anomalous regions (mask=1 -> bad)
        noise_mask = noise_mask * (1 - mask)

        # shape of noise is [B * 2, 1, H, W]
        perlin_mask = self.generate_perlin(b * 2, h, w).to(features.device)
        # only apply where perlin mask is 1
        noise_mask = noise_mask * perlin_mask

        # update gt mask
        mask = mask + noise_mask
        # binarize
        mask = torch.where(mask > 0, torch.ones_like(mask), torch.zeros_like(mask))

        # make new labels. 1 if any part of mask is 1, 0 otherwise
        new_anomalous = noise_mask.reshape(b * 2, -1).any(dim=1).type(torch.float32)
        labels = labels + new_anomalous
        # binarize
        labels = torch.where(labels > 0, torch.ones_like(labels), torch.zeros_like(labels))

        # apply masked noise
        perturbed = features + noise * noise_mask

        return perturbed, mask, labels
