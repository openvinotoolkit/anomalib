"""Validate torch image data.

Sections:
    - Item-level image validation
    - Batch-level image validation
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision.transforms.v2.functional import to_dtype_image
from torchvision.tv_tensors import Image


# Item-level image validation
def validate_dimensions(data: torch.Tensor, expected_dims: int) -> torch.Tensor:
    """Validate and correct the dimensions and channel order of the PyTorch tensor.

    Args:
        data: The input PyTorch tensor to validate.
        expected_dims: The expected number of dimensions.

    Returns:
        The validated and potentially corrected PyTorch tensor.

    Raises:
        ValueError: If the number of dimensions does not match the expected value
                    or if the number of channels is invalid.

    Examples:
        >>> import torch
        >>> from anomalib.data.io.validate import validate_torch_dimensions

        >>> # Correct 3D tensor (CHW)
        >>> tensor_chw = torch.rand(3, 100, 100)
        >>> result = validate_torch_dimensions(tensor_chw, 3)
        >>> result.shape
        torch.Size([3, 100, 100])

        >>> # 3D tensor in HWC format (will be corrected to CHW)
        >>> tensor_hwc = torch.rand(100, 100, 3)
        >>> result = validate_torch_dimensions(tensor_hwc, 3)
        >>> result.shape
        torch.Size([3, 100, 100])

        >>> # Invalid number of dimensions
        >>> tensor_2d = torch.rand(100, 100)
        >>> validate_torch_dimensions(tensor_2d, 3)
        Traceback (most recent call last):
            ...
        ValueError: Expected 3D data, got 2D data.

        >>> # Invalid number of channels
        >>> tensor_invalid = torch.rand(5, 100, 100)
        >>> validate_torch_dimensions(tensor_invalid, 3)
        Traceback (most recent call last):
            ...
        ValueError: Invalid number of channels: 5. Expected 1, 3, or 4.
    """
    if data.dim() != expected_dims:
        msg = f"Expected {expected_dims}D data, got {data.dim()}D data."
        raise ValueError(msg)

    if expected_dims == 3:
        if data.shape[-1] in {1, 3, 4} and data.shape[0] not in {1, 3, 4}:
            # Data is in HWC format, convert to CHW
            data = data.permute(2, 0, 1)
        elif data.shape[0] not in {1, 3, 4}:
            msg = f"Invalid number of channels: {data.shape[0]}. Expected 1, 3, or 4."
            raise ValueError(msg)

    return data


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


# Batch-level image validation
def validate_batch_image(image: torch.Tensor) -> Image:
    """Validate and convert the input PyTorch tensor image or batch of images.

    This function checks if the input is a valid image tensor and converts it to
    a torchvision Image object. It accepts both single images and batches of images.

    Args:
        image: The input image(s). Should be a PyTorch tensor with shape
               [C, H, W] for a single image or [N, C, H, W] for a batch of images.

    Returns:
        A torchvision Image object of the validated image(s).

    Raises:
        TypeError: If the input is not a PyTorch tensor.
        ValueError: If the input shape or number of channels is invalid.

    Examples:
        >>> import torch
        >>> # Single image
        >>> single_image = torch.rand(3, 224, 224)
        >>> result = validate_batch_image(single_image)
        >>> result.shape
        torch.Size([1, 3, 224, 224])

        >>> # Batch of images
        >>> batch_images = torch.rand(32, 3, 224, 224)
        >>> result = validate_batch_image(batch_images)
        >>> result.shape
        torch.Size([32, 3, 224, 224])

        >>> # Invalid number of channels
        >>> validate_batch_image(torch.rand(5, 224, 224))
        Traceback (most recent call last):
            ...
        ValueError: Invalid number of channels: 5. Expected 1, 3, or 4.
    """
    if not isinstance(image, torch.Tensor):
        msg = f"Image must be a torch.Tensor, got {type(image)}."
        raise TypeError(msg)
    if image.ndim not in {3, 4}:
        msg = f"Image must have shape [C, H, W] or [N, C, H, W], got shape {image.shape}."
        raise ValueError(msg)
    if image.ndim == 3:
        image = image.unsqueeze(0)  # add batch dimension
    if image.shape[1] not in {1, 3, 4}:
        msg = f"Invalid number of channels: {image.shape[1]}. Expected 1, 3, or 4."
        raise ValueError(msg)
    return Image(image)
