"""Numpy.ndarray validation functions for image data.

Sections:
    - Item-level image validation
    - Batch-level image validation
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np


# Item-level image validation
def validate_dimensions(data: np.ndarray, expected_dims: int) -> np.ndarray:
    """Validate and correct the dimensions and channel order of the numpy array.

    Args:
        data: The input numpy array to validate.
        expected_dims: The expected number of dimensions.

    Returns:
        The validated and potentially corrected numpy array.

    Raises:
        ValueError: If the number of dimensions does not match the expected value
                    or if the number of channels is invalid.

    Examples:
        >>> import numpy as np
        >>> from anomalib.data.io.validate import validate_numpy_dimensions

        >>> # Correct 3D array (HWC)
        >>> arr_hwc = np.random.rand(100, 100, 3)
        >>> result = validate_numpy_dimensions(arr_hwc, 3)
        >>> result.shape
        (100, 100, 3)

        >>> # 3D array in CHW format (will be corrected to HWC)
        >>> arr_chw = np.random.rand(3, 100, 100)
        >>> result = validate_numpy_dimensions(arr_chw, 3)
        >>> result.shape
        (100, 100, 3)

        >>> # Invalid number of dimensions
        >>> arr_2d = np.random.rand(100, 100)
        >>> validate_numpy_dimensions(arr_2d, 3)
        Traceback (most recent call last):
            ...
        ValueError: Expected 3D data, got 2D data.

        >>> # Invalid number of channels
        >>> arr_invalid = np.random.rand(100, 100, 5)
        >>> validate_numpy_dimensions(arr_invalid, 3)
        Traceback (most recent call last):
            ...
        ValueError: Invalid number of channels: 5. Expected 1, 3, or 4.
    """
    if data.ndim != expected_dims:
        msg = f"Expected {expected_dims}D data, got {data.ndim}D data."
        raise ValueError(msg)

    if expected_dims == 3:
        if data.shape[0] in {1, 3, 4} and data.shape[-1] not in {1, 3, 4}:
            # Data is in CHW format, convert to HWC
            data = np.transpose(data, (1, 2, 0))
        elif data.shape[-1] not in {1, 3, 4}:
            msg = f"Invalid number of channels: {data.shape[-1]}. Expected 1, 3, or 4."
            raise ValueError(msg)

    return data


# Batch-level image validation
def validate_batch_image(image: np.ndarray) -> np.ndarray:
    """Validate and convert the input NumPy array image or batch of images.

    This function checks if the input is a valid image array and ensures it has
    the correct shape and number of channels. It accepts both single images and
    batches of images.

    Args:
        image: The input image(s). Should be a NumPy array with shape
               [H, W, C] for a single image or [N, H, W, C] for a batch of images.

    Returns:
        A NumPy array of the validated image(s).

    Raises:
        TypeError: If the input is not a NumPy array.
        ValueError: If the input shape or number of channels is invalid.

    Examples:
        >>> import numpy as np
        >>> # Single image
        >>> single_image = np.random.rand(224, 224, 3)
        >>> result = validate_batch_image(single_image)
        >>> result.shape
        (1, 224, 224, 3)

        >>> # Batch of images
        >>> batch_images = np.random.rand(32, 224, 224, 3)
        >>> result = validate_batch_image(batch_images)
        >>> result.shape
        (32, 224, 224, 3)

        >>> # Invalid number of channels
        >>> validate_batch_image(np.random.rand(224, 224, 5))
        Traceback (most recent call last):
            ...
        ValueError: Invalid number of channels: 5. Expected 1, 3, or 4.
    """
    if not isinstance(image, np.ndarray):
        msg = f"Image must be a np.ndarray, got {type(image)}."
        raise TypeError(msg)
    if image.ndim not in {3, 4}:
        msg = f"Image must have shape [H, W, C] or [N, H, W, C], got shape {image.shape}."
        raise ValueError(msg)
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)  # add batch dimension
    if image.shape[-1] not in {1, 3, 4}:
        msg = f"Invalid number of channels: {image.shape[-1]}. Expected 1, 3, or 4."
        raise ValueError(msg)
    return image
