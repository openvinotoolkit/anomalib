"""Common validation functions for numpy.ndarray."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np


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


def validate_label(label: int | np.ndarray) -> np.ndarray:
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


def validate_anomaly_map(anomaly_map: np.ndarray) -> np.ndarray:
    """Validate and convert the input NumPy anomaly map.

    Args:
        anomaly_map: The input anomaly map to validate. Must be a NumPy array.

    Returns:
        The validated anomaly map as a NumPy array.

    Raises:
        TypeError: If the input is not a NumPy array.
        ValueError: If the anomaly map dimensions are invalid.

    Examples:
        >>> import numpy as np
        >>> from anomalib.data.io.validate import validate_numpy_anomaly_map

        >>> # 2D input
        >>> numpy_map = np.random.rand(100, 100)
        >>> result = validate_numpy_anomaly_map(numpy_map)
        >>> isinstance(result, np.ndarray)
        True
        >>> result.shape
        (100, 100)

        >>> # 3D input (will be squeezed)
        >>> numpy_map_3d = np.random.rand(1, 100, 100)
        >>> result = validate_numpy_anomaly_map(numpy_map_3d)
        >>> result.shape
        (100, 100)

        >>> # Invalid input
        >>> validate_numpy_anomaly_map(np.random.rand(3, 100, 100))
        Traceback (most recent call last):
            ...
        ValueError: Anomaly map must have 1 channel, got 3.
    """
    if not isinstance(anomaly_map, np.ndarray):
        msg = f"Anomaly map must be a np.ndarray, got {type(anomaly_map)}."
        raise TypeError(msg)

    if anomaly_map.ndim not in {2, 3}:
        msg = f"Anomaly map must have shape [H, W] or [1, H, W], got shape {anomaly_map.shape}."
        raise ValueError(msg)
    if anomaly_map.ndim == 3:
        if anomaly_map.shape[0] != 1:
            msg = f"Anomaly map must have 1 channel, got {anomaly_map.shape[0]}."
            raise ValueError(msg)
        anomaly_map = np.squeeze(anomaly_map, axis=0)
    return anomaly_map.astype(np.float32)


def validate_pred_score(pred_score: np.ndarray | float) -> np.ndarray:
    """Validate and convert the input NumPy prediction score.

    Args:
        pred_score: The input prediction score to validate. Can be a NumPy array or float.

    Returns:
        The validated prediction score as a NumPy array.

    Raises:
        TypeError: If the input is not a NumPy array or float.
        ValueError: If the prediction score is not a scalar.

    Examples:
        >>> import numpy as np
        >>> from anomalib.data.io.validate import validate_numpy_pred_score

        >>> # Scalar array input
        >>> numpy_score = np.array(0.8)
        >>> result = validate_numpy_pred_score(numpy_score)
        >>> isinstance(result, np.ndarray) and result.dtype == np.float32
        True
        >>> result.item()
        0.8

        >>> # Float input
        >>> result = validate_numpy_pred_score(0.7)
        >>> isinstance(result, np.ndarray) and result.dtype == np.float32
        True
        >>> result.item()
        0.7

        >>> # Invalid input
        >>> validate_numpy_pred_score(np.array([0.8, 0.9]))
        Traceback (most recent call last):
            ...
        ValueError: Prediction score must be a scalar, got shape (2,).
    """
    if isinstance(pred_score, float):
        pred_score = np.array(pred_score)
    if not isinstance(pred_score, np.ndarray):
        msg = f"Prediction score must be a np.ndarray or float, got {type(pred_score)}."
        raise TypeError(msg)
    pred_score = np.squeeze(pred_score)
    if pred_score.ndim != 0:
        msg = f"Prediction score must be a scalar, got shape {pred_score.shape}."
        raise ValueError(msg)
    return pred_score.astype(np.float32)


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
