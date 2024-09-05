"""Image validation functions."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision.transforms.v2.functional import to_dtype_image
from torchvision.tv_tensors import Image

from .common import validate_dimensions


def validate_image(image: torch.Tensor) -> Image:
    """Validate, correct, and scale the PyTorch tensor image.

    Args:
        image: The input PyTorch tensor image to validate.

    Returns:
        The validated, potentially corrected, and scaled PyTorch tensor image as an Image object.

    Raises:
        TypeError: If the input is not a PyTorch tensor.
        ValueError: If the number of dimensions is not 3 or if the number of channels is invalid.

    Examples:
        >>> import torch
        >>> from anomalib.data.io.validate import validate_torch_image

        >>> # Correct 3D tensor (CHW)
        >>> tensor_chw = torch.rand(3, 100, 100)
        >>> result = validate_torch_image(tensor_chw)
        >>> result.shape
        torch.Size([3, 100, 100])

        >>> # 3D tensor in HWC format (will be corrected to CHW)
        >>> tensor_hwc = torch.rand(100, 100, 3)
        >>> result = validate_torch_image(tensor_hwc)
        >>> result.shape
        torch.Size([3, 100, 100])

        >>> # Invalid input
        >>> validate_torch_image(torch.rand(5, 100, 100))
        Traceback (most recent call last):
            ...
        ValueError: Invalid number of channels: 5. Expected 1, 3, or 4.
    """
    if not isinstance(image, torch.Tensor):
        msg = f"Image must be a torch.Tensor, got {type(image)}."
        raise TypeError(msg)

    # Use validate_torch_dimensions to check and correct dimensions
    validated_image = validate_dimensions(image, expected_dims=3)

    # Use to_dtype_image for type conversion and scaling
    scaled_image = to_dtype_image(validated_image, dtype=torch.float32, scale=True)

    return Image(scaled_image)
