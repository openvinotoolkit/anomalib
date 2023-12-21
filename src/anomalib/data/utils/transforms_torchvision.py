"""Helper function for retrieving transforms."""

# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from enum import Enum

import torch
from torchvision.transforms import v2

from anomalib.data.utils.image import get_image_height_and_width

logger = logging.getLogger(__name__)


NORMALIZATION_STATS = {
    "imagenet": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "clip": {"mean": (0.48145466, 0.4578275, 0.40821073), "std": (0.26862954, 0.26130258, 0.27577711)},
}


class InputNormalizationMethod(str, Enum):
    """Normalization method for the input images."""

    NONE = "none"  # no normalization applied
    IMAGENET = "imagenet"  # normalization to ImageNet statistics
    CLIP = "clip"  # normalization to CLIP statistics


def get_transforms(
    config: str | v2.Compose | None = None,
    image_size: int | tuple[int, int] | None = None,
    center_crop: int | tuple[int, int] | None = None,
    normalization: InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
    _to_tensor: bool = True,
) -> v2.Compose:
    """Get transforms from config or image size.

    Args:
        config (str | v2.Compose | None, optional):
            Torchvision transforms.
            Either config or torchvision ``Compose`` object. Defaults to None.
        image_size (int | tuple | None, optional):
            Image size to transform.
            Defaults to None.
        center_crop (int | tuple | None, optional):
            Center crop size.
            Defaults to None.
        normalization (InputNormalizationMethod, optional):
            Normalization method for the input images.
            Defaults to InputNormalizationMethod.IMAGENET.
        _to_tensor (bool, optional):
            Boolean to convert the final transforms into Torch tensor.
            Defaults to True.

    Returns:
        T.Compose: Torchvision Compose object containing the image transforms.
    """
    if config is not None:
        # Load torchvision transforms from config
        pass  # Implement your logic for loading torchvision transforms from a config
    else:
        logger.info("No config file has been provided. Using default transforms.")
        transforms_list = []

        # Add resize transform
        if image_size is None:
            msg = "Both config and image_size cannot be `None`."
            raise ValueError(msg)
        resize_height, resize_width = get_image_height_and_width(image_size)
        transforms_list.append(v2.Resize(size=(resize_height, resize_width)))

        # Add center crop transform
        if center_crop is not None:
            crop_height, crop_width = get_image_height_and_width(center_crop)
            if crop_height > resize_height or crop_width > resize_width:
                msg = f"Crop size may not be larger than image size. Found {image_size} and {center_crop}"
                raise ValueError(msg)
            transforms_list.append(v2.CenterCrop(size=(crop_height, crop_width)))

        # Add convert-to-float transform
        transforms_list.append(v2.ToDtype(dtype=torch.float32, scale=True))

        # Add normalize transform
        if normalization in [InputNormalizationMethod.IMAGENET, InputNormalizationMethod.CLIP]:
            transforms_list.append(v2.Normalize(**NORMALIZATION_STATS[normalization]))
        elif normalization != InputNormalizationMethod.NONE:
            msg = f"Unknown normalization method: {normalization}"
            raise ValueError(msg)

        transforms = v2.Compose(transforms_list)

    return transforms
