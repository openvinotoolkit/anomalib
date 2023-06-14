"""Anomaly generator for the DSR model implementation."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from torch import Tensor, nn, F, randint, rand, from_numpy, zeros, stack
import random

import imgaug.augmenters as iaa
import numpy as np

from anomalib.data.utils.generators.perlin import _rand_perlin_2d_np

class DsrAnomalyGenerator(nn.Module):
    """Anomaly generator of the DSR model. The anomaly is generated using a Perlin
    noise generator on the two quantized representations of an image. This generator
    is only used during the second phase of training! The third phase requires generating
    smudges over the input images.
    """
    def __init__(
        self,
        p_anomalous: float = 0.5,
        beta: float | tuple[float, float] = (0.2, 1.0),
    ):
        self.p_anomalous = p_anomalous
        self.beta = beta

        self.augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45)),
        ]
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def transform_image(self, image):
        do_aug_orig = rand(1).numpy()[0] > 0.6
        if do_aug_orig:
            image = self.rot(image=image)
        image = image / 255.0
        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        return image


    def generate_anomaly(self, resize_shape: tuple[int, int]) -> Tensor:
        min_perlin_scale = 0
        perlin_scale = 6
        perlin_scalex = 2 ** (randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        threshold = 0.5
        perlin_noise_np = _rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]),
                                            (perlin_scalex, perlin_scaley))
        perlin_noise_np = self.rot(image=perlin_noise_np)
        mask = np.where(perlin_noise_np > threshold, np.ones_like(perlin_noise_np),
                              np.zeros_like(perlin_noise_np))
        mask = np.expand_dims(mask, axis=2).astype(np.float32)
        mask = from_numpy(mask)

        return mask
    
    def augment_batch(self, batch: Tensor, resize_shape: tuple[int, int]) -> tuple[Tensor, Tensor]:
        """Generate anomalous augmentations for a batch of input images.

        Args:
            batch (Tensor): Batch of input images

        Returns:
            Ground truth masks corresponding to the anomalous perturbations.
        """
        batch_size, _, height, width = batch.shape

        # Collect perturbations
        masks_list = []
        for _ in range(batch_size):
            if rand(1) > self.p_anomalous:  # include normal samples
                masks_list.append(zeros((1, height, width)))
            else:
                mask = self.generate_anomaly(resize_shape)
                masks_list.append(mask.permute((2, 0, 1)))

        masks = stack(masks_list).to(batch.device)

        return masks
