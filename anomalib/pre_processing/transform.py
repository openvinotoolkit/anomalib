"""Helper function for retrieving transforms."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional, Tuple, Union

import albumentations as A
from albumentations.pytorch import ToTensorV2

from anomalib.data.utils import get_image_height_and_width

logger = logging.getLogger(__name__)


def get_transforms(
    config: Optional[Union[str, A.Compose]] = None,
    image_size: Optional[Union[int, Tuple]] = None,
    center_crop: Optional[Union[int, Tuple]] = None,
    normalize: bool = True,
    to_tensor: bool = True,
) -> A.Compose:
    """Get transforms from config or image size.

    Args:
        config (Optional[Union[str, A.Compose]], optional): Albumentations transforms.
            Either config or albumentations ``Compose`` object. Defaults to None.
        image_size (Optional[Union[int, Tuple]], optional): Image size to transform. Defaults to None.
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
    assert config is not None or image_size is not None

    transforms: A.Compose

    if config is not None:
        # load transforms from config file
        if isinstance(config, str):
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
        if image_size is not None:
            height, width = get_image_height_and_width(image_size)
            transforms_list.append(A.Resize(height=height, width=width, always_apply=True))
        if center_crop is not None:
            height, width = get_image_height_and_width(center_crop)
            transforms_list.append(A.CenterCrop(height=height, width=width, always_apply=True))
        if normalize:
            transforms_list.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        else:
            transforms_list.append(A.ToFloat())
        if to_tensor:
            transforms_list.append(ToTensorV2())

        transforms = A.Compose(transforms_list)

    return transforms
