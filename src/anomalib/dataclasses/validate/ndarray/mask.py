"""Numpy.ndarray validation functions for mask data.

Sections:
    - Item-level mask validation
    - Batch-level mask validation
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np


# Item-level mask validation
def validate_mask(mask: np.ndarray) -> np.ndarray:
    """Validate and convert the input NumPy mask.

    Args:
        mask: The input mask to validate. Must be a NumPy array.

    Returns:
        The validated mask as a NumPy array.

    Raises:
        TypeError: If the input is not a NumPy array.
        ValueError: If the mask dimensions are invalid.

    Examples:
        >>> import numpy as np
        >>> from anomalib.data.io.validate import validate_numpy_mask

        >>> # 2D input
        >>> numpy_mask = np.random.randint(0, 2, (100, 100))
        >>> result = validate_numpy_mask(numpy_mask)
        >>> isinstance(result, np.ndarray)
        True
        >>> result.shape
        (100, 100)

        >>> # 3D input (will be squeezed)
        >>> numpy_mask_3d = np.random.randint(0, 2, (1, 100, 100))
        >>> result = validate_numpy_mask(numpy_mask_3d)
        >>> result.shape
        (100, 100)

        >>> # Invalid input
        >>> validate_numpy_mask(np.random.randint(0, 2, (3, 100, 100)))
        Traceback (most recent call last):
            ...
        alueError: Ground truth mask must have 1 channel, got 3.

        >>> validate_numpy_mask(torch.randint(0, 2, (100, 100)))
        Traceback (most recent call last):
            ...
        TypeError: Ground truth mask must be a np.ndarray, got <class 'torch.Tensor'>.
    """
    if not isinstance(mask, np.ndarray):
        msg = f"Ground truth mask must be a np.ndarray, got {type(mask)}."
        raise TypeError(msg)

    if mask.ndim not in {2, 3}:
        msg = f"Ground truth mask must have shape [H, W] or [1, H, W], got shape {mask.shape}."
        raise ValueError(msg)
    if mask.ndim == 3:
        if mask.shape[0] != 1:
            msg = f"Ground truth mask must have 1 channel, got {mask.shape[0]}."
            raise ValueError(msg)
        mask = np.squeeze(mask, axis=0)
    return mask.astype(bool)


def validate_pred_mask(pred_mask: np.ndarray) -> np.ndarray:
    """Validate and convert the input NumPy predicted mask.

    Args:
        pred_mask: The input predicted mask to validate. Must be a NumPy array.

    Returns:
        The validated predicted mask as a NumPy array.

    Raises:
        TypeError: If the input is not a NumPy array.
        ValueError: If the mask dimensions are invalid.

    Examples:
        >>> import numpy as np
        >>> from anomalib.data.io.validate import validate_numpy_pred_mask

        >>> # 2D input
        >>> numpy_mask = np.random.randint(0, 2, (100, 100))
        >>> result = validate_numpy_pred_mask(numpy_mask)
        >>> isinstance(result, np.ndarray)
        True
        >>> result.shape
        (100, 100)

        >>> # 3D input (will be squeezed)
        >>> numpy_mask_3d = np.random.randint(0, 2, (1, 100, 100))
        >>> result = validate_numpy_pred_mask(numpy_mask_3d)
        >>> result.shape
        (100, 100)

        >>> # Invalid input
        >>> validate_numpy_pred_mask(np.random.randint(0, 2, (3, 100, 100)))
        Traceback (most recent call last):
            ...
        ValueError: Predicted mask must have 1 channel, got 3.
    """
    if not isinstance(pred_mask, np.ndarray):
        msg = f"Predicted mask must be a np.ndarray, got {type(pred_mask)}."
        raise TypeError(msg)

    if pred_mask.ndim not in {2, 3}:
        msg = f"Predicted mask must have shape [H, W] or [1, H, W], got shape {pred_mask.shape}."
        raise ValueError(msg)
    if pred_mask.ndim == 3:
        if pred_mask.shape[0] != 1:
            msg = f"Predicted mask must have 1 channel, got {pred_mask.shape[0]}."
            raise ValueError(msg)
        pred_mask = np.squeeze(pred_mask, axis=0)
    return pred_mask.astype(bool)


# Batch-level mask validation
def validate_batch_mask(gt_mask: np.ndarray | None, batch_size: int) -> np.ndarray | None:
    """Validate and convert the input batch of numpy masks.

    Args:
        gt_mask: The input ground truth masks. Can be a numpy array or None.
        batch_size: The expected batch size.

    Returns:
        A numpy array of validated masks, or None if the input was None.

    Raises:
        TypeError: If the input is not a numpy array.
        ValueError: If the input shape is invalid.

    Examples:
        >>> import numpy as np
        >>> mask = np.random.randint(0, 2, (4, 224, 224))
        >>> result = validate_batch_mask(mask, 4)
        >>> result.shape
        (4, 224, 224)

        >>> validate_batch_mask(None, 4)
        None

        >>> validate_batch_mask(np.random.rand(4, 3, 224, 224), 4)
        Traceback (most recent call last):
            ...
        ValueError: Ground truth mask must have 1 channel, got 3.
    """
    if gt_mask is None:
        return None
    if not isinstance(gt_mask, np.ndarray):
        msg = f"Ground truth mask must be a np.ndarray, got {type(gt_mask)}."
        raise TypeError(msg)
    if gt_mask.ndim not in {2, 3, 4}:
        msg = f"Ground truth mask must have shape [H, W] or [N, H, W] or [N, 1, H, W], got shape {gt_mask.shape}."
        raise ValueError(msg)
    if gt_mask.ndim == 2:
        if batch_size != 1:
            msg = f"Invalid shape for gt_mask. Got mask shape {gt_mask.shape} for batch size {batch_size}."
            raise ValueError(msg)
        gt_mask = gt_mask[np.newaxis, ...]
    if gt_mask.ndim == 3 and gt_mask.shape[0] != batch_size:
        msg = f"Invalid shape for gt_mask. Got mask shape {gt_mask.shape} for batch size {batch_size}."
        raise ValueError(msg)
    if gt_mask.ndim == 4:
        if gt_mask.shape[1] != 1:
            msg = f"Ground truth mask must have 1 channel, got {gt_mask.shape[1]}."
            raise ValueError(msg)
        gt_mask = np.squeeze(gt_mask, axis=1)
    return gt_mask.astype(bool)


def validate_batch_pred_mask(pred_mask: np.ndarray | None, batch_size: int) -> np.ndarray | None:
    """Validate and convert the input batch of numpy predicted masks.

    Args:
        pred_mask: The input predicted masks. Can be a numpy array or None.
        batch_size: The expected batch size.

    Returns:
        A numpy array of validated predicted masks, or None if the input was None.

    Raises:
        TypeError: If the input is not a numpy array.
        ValueError: If the input shape is invalid.

    Examples:
        >>> import numpy as np
        >>> pred_mask = np.random.randint(0, 2, (4, 224, 224))
        >>> result = validate_batch_pred_mask(pred_mask, 4)
        >>> result.shape
        (4, 224, 224)

        >>> validate_batch_pred_mask(None, 4)
        None

        >>> validate_batch_pred_mask(np.random.rand(4, 3, 224, 224), 4)
        Traceback (most recent call last):
            ...
        ValueError: Predicted mask must have 1 channel, got 3.
    """
    if pred_mask is None:
        return None
    if not isinstance(pred_mask, np.ndarray):
        msg = f"Predicted mask must be a np.ndarray, got {type(pred_mask)}."
        raise TypeError(msg)
    if pred_mask.ndim not in {2, 3, 4}:
        msg = f"Predicted mask must have shape [H, W] or [N, H, W] or [N, 1, H, W], got shape {pred_mask.shape}."
        raise ValueError(msg)
    if pred_mask.ndim == 2:
        if batch_size != 1:
            msg = f"Invalid shape for pred_mask. Got shape {pred_mask.shape} for batch size {batch_size}."
            raise ValueError(msg)
        pred_mask = pred_mask[np.newaxis, ...]
    if pred_mask.ndim == 4:
        if pred_mask.shape[1] != 1:
            msg = f"Predicted mask must have 1 channel, got {pred_mask.shape[1]}."
            raise ValueError(msg)
        pred_mask = np.squeeze(pred_mask, axis=1)
    return pred_mask.astype(bool)
