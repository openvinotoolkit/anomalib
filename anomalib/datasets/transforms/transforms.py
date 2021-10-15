"""
Anomalib Transforms via Albumentations
"""

from typing import Union

import albumentations as a
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, ListConfig

from .custom_transforms import BilateralFilter, RgbToGray


def get_transforms(config: Union[DictConfig, ListConfig], is_train: bool = True, to_tensor: bool = True) -> a.Compose:
    """
    Build a pipeline of image transforms each with different probability defined in config file

    Args:
        config: Transformation configurations
        is_train: Training or Inferencer phase. For training, a probability is passed to the list
            of transformations.
        to_tensor: Boolean to convert the list of transforms to tensor. This is to be `True` if
            training or inference is done with `PyTorch`. For `OpenVINO` Inferencer, this is to set
            to `False`.

    Returns:
        Composed augmentation pipeline

    """
    crop_size = config.image_size if config.crop_size is None else config.crop_size
    transforms = [
        a.Resize(config.image_size[0], config.image_size[1], always_apply=True),
        BilateralFilter(
            diameter=config.bilateral_filter.diameter,
            sigma_color=config.bilateral_filter.sigma_color,
            sigma_space=config.bilateral_filter.sigma_space,
            always_apply=config.bilateral_filter.always_apply,
        ),
        a.OneOf(
            [
                a.RandomRotate90(),
                a.HorizontalFlip(),
            ],
            p=config.rotate_flip_p,
        ),
        a.GaussNoise(p=config.gauss_noise_p),
        a.OneOf(
            [
                a.MotionBlur(p=config.blur.motion_blur_p),
                a.MedianBlur(blur_limit=config.blur.median_blur.blur_limit, p=config.blur.median_blur.p),
                a.Blur(blur_limit=config.blur.blur.blur_limit, p=config.blur.blur.p),
            ],
            p=config.blur.p,
        ),
        a.ShiftScaleRotate(
            shift_limit=config.shift_scale_rotate.shift_limit,
            scale_limit=config.shift_scale_rotate.scale_limit,
            rotate_limit=config.shift_scale_rotate.rotate_limit,
            p=config.shift_scale_rotate.p,
        ),
        a.OneOf(
            [
                a.OpticalDistortion(p=config.geometric_transforms.optical_distortion_p),
                a.GridDistortion(p=config.geometric_transforms.grid_distortion_p),
                a.PiecewiseAffine(p=config.geometric_transforms.affine_p),
            ],
            p=config.geometric_transforms.p,
        ),
        a.OneOf(
            [
                a.CLAHE(clip_limit=config.image_adjustments.clahe_clip_limit),
                a.Sharpen(),
                a.Emboss(),
                a.RandomBrightnessContrast(),
                a.HueSaturationValue(),
            ],
            p=config.image_adjustments.p,
        ),
        a.ImageCompression(
            p=config.image_compression.p,
            quality_lower=config.image_compression.quality_lower,
            quality_upper=config.image_compression.quality_upper,
        ),
        a.CenterCrop(crop_size[0], crop_size[1], always_apply=True),
        a.Normalize(mean=config.normalize.mean, std=config.normalize.std, always_apply=True),
        RgbToGray(always_apply=config.grayscale),
    ]

    if to_tensor:
        transforms.append(ToTensorV2(always_apply=True))

    transforms = a.Compose(
        transforms=transforms,
        p=config.p if is_train else 0.0,
    )

    return transforms
