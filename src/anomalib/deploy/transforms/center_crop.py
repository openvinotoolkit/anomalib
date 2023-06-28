"""Overrides CenterCrop from torchvision to avoid ``__round__`` not defined on Tensor"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torchvision.transforms.functional import crop, get_dimensions, pad
from torchvision.transforms.transforms import _setup_size


class CenterCrop(nn.Module):
    """CenterCrop module to crop the image from the center.

    Note: This is slightly modified from F.CenterCrop
    """

    def __init__(self, size):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

    def forward(self, img):
        _, image_height, image_width = get_dimensions(img)
        crop_height, crop_width = self.size
        if crop_width > image_width or crop_height > image_height:
            padding_ltrb = [
                (crop_width - image_width) // 2 if crop_width > image_width else 0,
                (crop_height - image_height) // 2 if crop_height > image_height else 0,
                (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
                (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
            ]
            img = pad(img, padding_ltrb, fill=0)  # PIL uses fill value 0
            _, image_height, image_width = get_dimensions(img)
            if crop_width == image_width and crop_height == image_height:
                return img

        crop_top = self.crop_dims(image_height, crop_height)
        crop_left = self.crop_dims(image_width, crop_width)
        return crop(img, crop_top, crop_left, crop_height, crop_width)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"

    @staticmethod
    def crop_dims(image_dim: int | torch.Tensor, crop_dim: int | torch.Tensor) -> int | torch.Tensor:
        value = (image_dim - crop_dim) / 2.0
        if isinstance(image_dim, torch.Tensor):
            value = torch.round(value).to(torch.int32)
        else:
            value = int(round(value))
        return value
