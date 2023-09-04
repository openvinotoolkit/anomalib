"""Helper function for retrieving transforms."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from enum import Enum

import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig

from anomalib.data.utils.image import get_image_height_and_width

logger = logging.getLogger(__name__)


class InputNormalizationMethod(str, Enum):
    """Normalization method for the input images."""

    NONE = "none"  # no normalization applied
    IMAGENET = "imagenet"  # normalization to ImageNet statistics


def get_transforms(
    config: str | A.Compose | None = None,
    image_size: int | tuple[int, int] | None = None,
    center_crop: int | tuple[int, int] | None = None,
    normalization: InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
    to_tensor: bool = True,
) -> A.Compose:
    """Get transforms from config or image size.

    Args:
        config (str | A.Compose | None, optional): Albumentations transforms.
            Either config or albumentations ``Compose`` object. Defaults to None.
        image_size (int | tuple | None, optional): Image size to transform. Defaults to None.
        to_tensor (bool, optional): Boolean to convert the final transforms into Torch tensor. Defaults to True.

    Raises:
        ValueError: When both ``config`` and ``image_size`` is ``None``.
        ValueError: When ``config`` is not a ``str`` or `A.Compose`` object.

    Returns:
        A.Compose: Albumentation ``Compose`` object containing the image transforms.

    Examples:
        >>> import skimage
        >>> image = skimage.data.astronaut()

        >>> transforms = get_transforms(image_size=256, to_tensor=False)
        >>> output = transforms(image=image)
        >>> output["image"].shape
        (256, 256, 3)

        >>> transforms = get_transforms(image_size=256, to_tensor=True)
        >>> output = transforms(image=image)
        >>> output["image"].shape
        torch.Size([3, 256, 256])


        Transforms could be read from albumentations Compose object.
        >>> import albumentations as A
        >>> from albumentations.pytorch import ToTensorV2
        >>> config = A.Compose([A.Resize(512, 512), ToTensorV2()])
        >>> transforms = get_transforms(config=config, to_tensor=False)
        >>> output = transforms(image=image)
        >>> output["image"].shape
        (512, 512, 3)
        >>> type(output["image"])
        numpy.ndarray

        Transforms could be deserialized from a yaml file.
        >>> transforms = A.Compose([A.Resize(1024, 1024), ToTensorV2()])
        >>> A.save(transforms, "/tmp/transforms.yaml", data_format="yaml")
        >>> transforms = get_transforms(config="/tmp/transforms.yaml")
        >>> output = transforms(image=image)
        >>> output["image"].shape
        torch.Size([3, 1024, 1024])
    """
    transforms: A.Compose

    if config is not None:
        if isinstance(config, DictConfig):
            logger.info("Loading transforms from config File")
            transforms_list = []
            for key, value in config.items():
                if hasattr(A, key):
                    transform = getattr(A, key)(**value)
                    logger.info(f"Transform {transform} added!")
                    transforms_list.append(transform)
                else:
                    raise ValueError(f"Transformation {key} is not part of albumentations")

            transforms_list.append(ToTensorV2())
            transforms = A.Compose(transforms_list, additional_targets={"image": "image", "depth_image": "image"})

        # load transforms from config file
        elif isinstance(config, str):
            logger.info("Reading transforms from Albumentations config file: %s.", config)
            transforms = A.load(filepath=config, data_format="yaml")
        elif isinstance(config, A.Compose):
            logger.info("Transforms loaded from Albumentations Compose object")
            transforms = config
        else:
            raise ValueError("config could be either ``str`` or ``A.Compose``")
    else:
        logger.info("No config file has been provided. Using default transforms.")
        transforms_list = []

        # add resize transform
        if image_size is None:
            raise ValueError(
                "Both config and image_size cannot be `None`. "
                "Provide either config file to de-serialize transforms "
                "or image_size to get the default transformations"
            )
        resize_height, resize_width = get_image_height_and_width(image_size)
        transforms_list.append(A.Resize(height=resize_height, width=resize_width, always_apply=True))

        # add center crop transform
        if center_crop is not None:
            crop_height, crop_width = get_image_height_and_width(center_crop)
            if crop_height > resize_height or crop_width > resize_width:
                raise ValueError(f"Crop size may not be larger than image size. Found {image_size} and {center_crop}")
            transforms_list.append(A.CenterCrop(height=crop_height, width=crop_width, always_apply=True))

        # add normalize transform
        if normalization == InputNormalizationMethod.IMAGENET:
            transforms_list.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        elif normalization == InputNormalizationMethod.NONE:
            transforms_list.append(A.ToFloat(max_value=255))
        else:
            raise ValueError(f"Unknown normalization method: {normalization}")

        # add tensor conversion
        if to_tensor:
            transforms_list.append(ToTensorV2())

        transforms = A.Compose(transforms_list, additional_targets={"image": "image", "depth_image": "image"})

    return transforms
