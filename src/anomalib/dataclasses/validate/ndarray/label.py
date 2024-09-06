"""Numpy.ndarray validation functions for label data.

Sections:
    - Item-level label validation
    - Batch-level label validation
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import numpy as np


# Item-level label validation
def validate_gt_label(label: int | np.ndarray) -> np.ndarray:
    """Validate and convert the input label to a boolean numpy array.

    Args:
        label: The input label to validate.

    Returns:
        The validated label as a boolean numpy array.

    Raises:
        TypeError: If the input is not an int or numpy array.

    Examples:
        >>> import numpy as np
        >>> from anomalib.data.io.validate import validate_numpy_label
        >>>
        >>> # Integer input
        >>> result = validate_numpy_label(1)
        >>> result
        array(True)

        >>> # Numpy array input
        >>> arr = np.array([0, 1, 1, 0])
        >>> result = validate_numpy_label(arr)
        >>> result
        array([False,  True,  True, False])

        >>> # Invalid input
        >>> validate_numpy_label("invalid")
        Traceback (most recent call last):
            ...
        TypeError: Unsupported label type: <class 'str'>. Expected int or numpy array.
    """
    if isinstance(label, int | np.integer):
        return np.array(label, dtype=bool)
    if isinstance(label, np.ndarray):
        return label.astype(bool)
    msg = f"Unsupported label type: {type(label)}. Expected int or numpy array."
    raise TypeError(msg)


def validate_pred_label(pred_label: np.ndarray | int) -> np.ndarray:
    """Validate and convert the input NumPy predicted label.

    Args:
        pred_label: The input predicted label to validate. Can be a NumPy array or int.

    Returns:
        The validated predicted label as a NumPy array.

    Raises:
        TypeError: If the input is not a NumPy array or int.
        ValueError: If the predicted label is not a scalar.

    Examples:
        >>> import numpy as np
        >>> from anomalib.data.io.validate import validate_numpy_pred_label

        >>> # Scalar array input
        >>> numpy_label = np.array(1)
        >>> result = validate_numpy_pred_label(numpy_label)
        >>> isinstance(result, np.ndarray) and result.dtype == bool
        True
        >>> result.item()
        True

        >>> # Integer input
        >>> result = validate_numpy_pred_label(0)
        >>> isinstance(result, np.ndarray) and result.dtype == bool
        True
        >>> result.item()
        False

        >>> # Invalid input
        >>> validate_numpy_pred_label(np.array([0, 1]))
        Traceback (most recent call last):
            ...
        ValueError: Predicted label must be a scalar, got shape (2,).
    """
    if isinstance(pred_label, int):
        pred_label = np.array(pred_label)
    if not isinstance(pred_label, np.ndarray):
        msg = f"Predicted label must be a np.ndarray or int, got {type(pred_label)}."
        raise TypeError(msg)
    pred_label = np.squeeze(pred_label)
    if pred_label.ndim != 0:
        msg = f"Predicted label must be a scalar, got shape {pred_label.shape}."
        raise ValueError(msg)
    return pred_label.astype(bool)


# Batch-level label validation
def validate_batch_label(gt_label: np.ndarray | Sequence[int] | None, batch_size: int) -> np.ndarray | None:
    """Validate and convert the input batch of labels to a boolean numpy array.

    Args:
        gt_label: The input ground truth labels. Can be a numpy array, a sequence of integers, or None.
        batch_size: The expected batch size.

    Returns:
        A boolean numpy array of validated labels, or None if the input was None.

    Raises:
        TypeError: If the input is not a numpy array or a sequence of integers.
        ValueError: If the input shape or type is invalid.

    Examples:
        >>> import numpy as np
        >>> validate_batch_label(np.array([0, 1, 1, 0]), 4)
        array([False,  True,  True, False])

        >>> validate_batch_label([0, 1, 1, 0], 4)
        array([False,  True,  True, False])

        >>> validate_batch_label(None, 4)
        None

        >>> validate_batch_label(np.array([0.5, 1.5]), 2)
        Traceback (most recent call last):
            ...
        ValueError: Ground truth label must be boolean or integer, got float64.
    """
    if gt_label is None:
        return None
    if isinstance(gt_label, Sequence):
        gt_label = np.array(gt_label)
    if not isinstance(gt_label, np.ndarray):
        msg = f"Ground truth label must be a sequence of integers or a np.ndarray, got {type(gt_label)}."
        raise TypeError(msg)
    if gt_label.ndim != 1:
        msg = f"Ground truth label must be a 1-dimensional vector, got shape {gt_label.shape}."
        raise ValueError(msg)
    if len(gt_label) != batch_size:
        msg = f"Ground truth label must have length {batch_size}, got length {len(gt_label)}."
        raise ValueError(msg)
    if not np.issubdtype(gt_label.dtype, np.integer) and not np.issubdtype(gt_label.dtype, bool):
        msg = f"Ground truth label must be boolean or integer, got {gt_label.dtype}."
        raise ValueError(msg)
    return gt_label.astype(bool)


def validate_batch_pred_label(pred_label: np.ndarray | None, batch_size: int) -> np.ndarray | None:
    """Validate and convert the input batch of numpy predicted labels.

    Args:
        pred_label: The input predicted labels. Can be a numpy array or None.
        batch_size: The expected batch size.

    Returns:
        A numpy array of validated predicted labels, or None if the input was None.

    Raises:
        TypeError: If the input is not a numpy array.
        ValueError: If the input shape is invalid.

    Examples:
        >>> import numpy as np
        >>> pred_label = np.random.randint(0, 2, (4,))
        >>> result = validate_batch_pred_label(pred_label, 4)
        >>> result.shape
        (4,)

        >>> validate_batch_pred_label(None, 4)
        None

        >>> validate_batch_pred_label(np.random.rand(4, 1), 4)
        Traceback (most recent call last):
            ...
        ValueError: Predicted label must be a 1-dimensional vector, got shape (4, 1).
    """
    if pred_label is None:
        return None
    if not isinstance(pred_label, np.ndarray):
        msg = f"Predicted label must be a np.ndarray, got {type(pred_label)}."
        raise TypeError(msg)
    pred_label = np.squeeze(pred_label)
    if pred_label.ndim != 1:
        msg = f"Predicted label must be a 1-dimensional vector, got shape {pred_label.shape}."
        raise ValueError(msg)
    if len(pred_label) != batch_size:
        msg = f"Predicted label must have length {batch_size}, got length {len(pred_label)}."
        raise ValueError(msg)
    return pred_label.astype(bool)
