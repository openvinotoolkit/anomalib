"""Custom Torchvision transforms for Anomalib."""

# Original Code
# Copyright (c) Soumith Chintala 2016
# https://github.com/pytorch/vision/blob/v0.16.1/torchvision/transforms/v2/functional/_geometry.py
# SPDX-License-Identifier: BSD-3-Clause
#
# Modified
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from torch.nn.functional import pad
from torchvision.transforms.v2 import Transform
from torchvision.transforms.v2.functional._geometry import (
    _center_crop_compute_padding,
    _center_crop_parse_output_size,
    _parse_pad_padding,
)


def _center_crop_compute_crop_anchor(
    crop_height: int,
    crop_width: int,
    image_height: int,
    image_width: int,
) -> tuple[int, int]:
    """Compute the anchor point for center-cropping.

    This function is a modified version of the torchvision.transforms.functional._center_crop_compute_crop_anchor
    function. The original function uses `round` to compute the anchor point, which is not compatible with ONNX.

    Args:
        crop_height (int): Desired height of the crop.
        crop_width (int): Desired width of the crop.
        image_height (int): Height of the input image.
        image_width (int): Width of the input image.
    """
    crop_top = torch.tensor((image_height - crop_height) / 2.0).round().int().item()
    crop_left = torch.tensor((image_width - crop_width) / 2.0).round().int().item()
    return crop_top, crop_left


def center_crop_image(image: torch.Tensor, output_size: list[int]) -> torch.Tensor:
    """Apply center-cropping to an input image.

    Uses the modified anchor point computation function to compute the anchor point for center-cropping.

    Args:
        image (torch.Tensor): Input image to be center-cropped.
        output_size (list[int]): Desired output size of the crop.
    """
    crop_height, crop_width = _center_crop_parse_output_size(output_size)
    shape = image.shape
    if image.numel() == 0:
        return image.reshape(shape[:-2] + (crop_height, crop_width))
    image_height, image_width = shape[-2:]

    if crop_height > image_height or crop_width > image_width:
        padding_ltrb = _center_crop_compute_padding(crop_height, crop_width, image_height, image_width)
        image = pad(image, _parse_pad_padding(padding_ltrb), value=0.0)

        image_height, image_width = image.shape[-2:]
        if crop_width == image_width and crop_height == image_height:
            return image

    crop_top, crop_left = _center_crop_compute_crop_anchor(crop_height, crop_width, image_height, image_width)
    return image[..., crop_top : (crop_top + crop_height), crop_left : (crop_left + crop_width)]


class ExportableCenterCrop(Transform):
    """Transform that applies center-cropping to an input image and allows to be exported to ONNX.

    Args:
        size (int | tuple[int, int]): Desired output size of the crop.
    """

    def __init__(self, size: int | tuple[int, int]) -> None:
        super().__init__()
        self.size = list(size) if isinstance(size, tuple) else [size, size]

    def _transform(self, inpt: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        """Apply the transform."""
        del params
        return center_crop_image(inpt, output_size=self.size)
