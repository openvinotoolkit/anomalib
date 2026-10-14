# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Few-shot training augmentation for FoundAD.

Ports ``build_train_transform_staged`` from the original FoundAD repo
(``foundad/src/datasets/dataset.py``), which the authors apply to the base
training images (before CutPaste) to compensate for having only a handful of
real training samples (1/2/4-shot). Without it, the manifold projector can
trivially overfit to the exact pixels of the few available images instead of
learning a general notion of the normal-image manifold.

The augmentation has two independent stages, each applied with its own
probability:
    1. **Orientation**: with probability ``p_orient``, randomly pick one of
       {horizontal flip, vertical flip, 90/270 degree rotation} and apply it.
    2. **Appearance**: with probability ``p_appear``, randomly pick one of
       {color jitter, grayscale, Gaussian blur} and apply it.

This matches the paper's default training config (``use_hflip``,
``use_vflip``, ``use_rotate90``, ``use_color_jitter``, ``use_gray``,
``use_blur`` are all enabled for both MVTec-AD and VisA few-shot training,
per ``foundad/configs/config.yaml``).

Reference:
    Zhai et al., "Foundation Visual Encoders Are Secretly Few-Shot Anomaly
    Detectors", ICLR 2026. https://arxiv.org/abs/2510.01934
"""

import random
from collections.abc import Callable  # noqa: TC003 - used at runtime in an instance-attribute annotation

import torch
import torchvision.transforms.functional as TF  # noqa: N812
from torchvision import transforms
from torchvision.transforms import v2

from anomalib.models.image.foundad.components.cutpaste import _apply_in_pixel_space


class FewShotAugmentation:
    """Orientation + appearance augmentation for FoundAD's few-shot training images.

    Applied to the (already ImageNet-normalized) base training batch before
    CutPaste, to expand the effective diversity of a tiny few-shot training
    set. Color jitter and grayscale are applied in real, un-normalized pixel
    space (reusing
    :func:`anomalib.models.image.foundad.components.cutpaste._apply_in_pixel_space`):
    color jitter to avoid the RGB->HSV division-by-zero hazard documented in
    ``cutpaste.py``, and grayscale because its fixed per-channel luminance
    weights assume equally-scaled RGB channels (each ImageNet channel has a
    different mean/std). Gaussian blur is a per-channel linear operation and
    is safe to apply directly to normalized tensors.

    Args:
        p_orient: Probability of applying an orientation change (flip/rotate).
            Defaults to ``0.3`` (matches the paper's default).
        p_appear: Probability of applying an appearance change (color
            jitter/grayscale/blur). Defaults to ``0.3`` (matches the paper's
            default).
        blur_kernel_size: Kernel size for Gaussian blur. Defaults to ``23``,
            matching the paper's ``23 if resize >= 384 else 11`` for FoundAD's
            default 518x518 image size.

    Example:
        >>> augment = FewShotAugmentation()
        >>> images = torch.randn(4, 3, 518, 518)
        >>> augmented = augment(images)
        >>> augmented.shape
        torch.Size([4, 3, 518, 518])
    """

    def __init__(
        self,
        p_orient: float = 0.3,
        p_appear: float = 0.3,
        blur_kernel_size: int = 23,
    ) -> None:
        self.p_orient = p_orient
        self.p_appear = p_appear
        self._orient_ops: list[Callable[[torch.Tensor], torch.Tensor]] = [
            TF.hflip,
            TF.vflip,
            v2.RandomChoice(
                [
                    v2.RandomRotation((90, 90), expand=True),
                    v2.RandomRotation((270, 270), expand=True),
                ],
            ),
        ]
        # Matches the paper's fixed ColorJitter(0.3, 0.3, 0.3, 0.05).
        self._color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
        self._blur = transforms.GaussianBlur(kernel_size=blur_kernel_size, sigma=(0.1, 2.0))

    def _appearance_op(self, img: torch.Tensor) -> torch.Tensor:
        """Apply one randomly chosen appearance-space augmentation to ``img``."""
        choice = random.randint(0, 2)  # noqa: S311  # nosec B311
        if choice == 0:
            return _apply_in_pixel_space(img, self._color_jitter)
        if choice == 1:
            return _apply_in_pixel_space(img, lambda x: TF.rgb_to_grayscale(x, num_output_channels=3))
        return self._blur(img)

    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        """Apply random orientation/appearance augmentation to a batch of images.

        Args:
            imgs: Batch of ImageNet-normalized images, shape (B, C, H, W).

        Returns:
            Augmented batch of the same shape.
        """
        augmented = imgs.clone()
        for i in range(imgs.shape[0]):
            img = augmented[i]
            if random.random() < self.p_orient:  # noqa: S311  # nosec B311
                img = random.choice(self._orient_ops)(img)  # noqa: S311  # nosec B311
            if random.random() < self.p_appear:  # noqa: S311  # nosec B311
                img = self._appearance_op(img)
            augmented[i] = img
        return augmented
