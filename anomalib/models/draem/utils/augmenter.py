"""Augmenter module to generates out-of-distribution samples for the DRAEM implementation."""

# Original Code
# Copyright (c) 2021 VitjanZ
# https://github.com/VitjanZ/DRAEM.
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import glob
import math
import random
from typing import Optional, Tuple

import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
from torch import Tensor
from torchvision.datasets.folder import IMG_EXTENSIONS

from anomalib.data.utils import random_2d_perlin


def nextpow2(value):
    """Returns the smallest power of 2 greater than or equal to the input value."""
    return 2 ** (math.ceil(math.log(value, 2)))


class Augmenter:
    """Class that generates noisy augmentations of input images.

    Args:
        anomaly_source_path (Optional[str]): Path to a folder of images that will be used as source of the anomalous
        noise. If not specified, random noise will be used instead.
    """

    def __init__(self, anomaly_source_path: Optional[str] = None):

        self.anomaly_source_paths = []
        if anomaly_source_path is not None:
            for img_ext in IMG_EXTENSIONS:
                self.anomaly_source_paths.extend(glob.glob(anomaly_source_path + "/**/*" + img_ext, recursive=True))

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

    def rand_augmenter(self) -> iaa.Sequential:
        """Selects 3 random transforms that will be applied to the anomaly source images.

        Returns:
            A selection of 3 transforms.
        """
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]], self.augmenters[aug_ind[1]], self.augmenters[aug_ind[2]]])
        return aug

    def generate_perturbation(
        self, height: int, width: int, anomaly_source_path: Optional[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate an image containing a random anomalous perturbation using a source image.

        Args:
            height (int): height of the generated image.
            width: (int): width of the generated image.
            anomaly_source_path (Optional[str]): Path to an image file. If not provided, random noise will be used
            instead.

        Returns:
            Image containing a random anomalous perturbation, and the corresponding ground truth anomaly mask.
        """
        # Generate random perlin noise
        perlin_scale = 6
        min_perlin_scale = 0

        perlin_scalex = 2 ** random.randint(min_perlin_scale, perlin_scale)
        perlin_scaley = 2 ** random.randint(min_perlin_scale, perlin_scale)

        perlin_noise = random_2d_perlin((nextpow2(height), nextpow2(width)), (perlin_scalex, perlin_scaley))[
            :height, :width
        ]
        perlin_noise = self.rot(image=perlin_noise)

        # Create mask from perlin noise
        mask = np.where(perlin_noise > 0.5, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        mask = np.expand_dims(mask, axis=2).astype(np.float32)

        # Load anomaly source image
        if anomaly_source_path:
            anomaly_source_img = cv2.imread(anomaly_source_path)
            anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(width, height))
        else:  # if no anomaly source is specified, we use the perlin noise as anomalous source
            anomaly_source_img = np.expand_dims(perlin_noise, 2).repeat(3, 2)
            anomaly_source_img = (anomaly_source_img * 255).astype(np.uint8)

        # Augment anomaly source image
        aug = self.rand_augmenter()
        anomaly_img_augmented = aug(image=anomaly_source_img)

        # Create anomalous perturbation that we will apply to the image
        perturbation = anomaly_img_augmented.astype(np.float32) * mask / 255.0

        return perturbation, mask

    def augment_batch(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        """Generate anomalous augmentations for a batch of input images.

        Args:
            batch (Tensor): Batch of input images

        Returns:
            - Augmented image to which anomalous perturbations have been added.
            - Ground truth masks corresponding to the anomalous perturbations.
        """
        batch_size, channels, height, width = batch.shape

        # Collect perturbations
        perturbations_list = []
        masks_list = []
        for _ in range(batch_size):
            if random.random() > 0.5:  # include 50% normal samples
                perturbations_list.append(torch.zeros((channels, height, width)))
                masks_list.append(torch.zeros((1, height, width)))
            else:
                anomaly_source_path = (
                    random.sample(self.anomaly_source_paths, 1)[0] if len(self.anomaly_source_paths) > 0 else None
                )
                perturbation, mask = self.generate_perturbation(height, width, anomaly_source_path)
                perturbations_list.append(Tensor(perturbation).permute((2, 0, 1)))
                masks_list.append(Tensor(mask).permute((2, 0, 1)))

        perturbations = torch.stack(perturbations_list).to(batch.device)
        masks = torch.stack(masks_list).to(batch.device)

        # Apply perturbations batch wise
        beta = torch.rand(batch_size) * 0.8
        beta = beta.view(batch_size, 1, 1, 1).expand_as(batch).to(batch.device)

        augmented_batch = batch * (1 - masks) + (1 - beta) * perturbations + beta * batch * (masks)

        return augmented_batch, masks
