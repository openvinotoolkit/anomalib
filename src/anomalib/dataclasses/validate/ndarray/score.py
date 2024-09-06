"""Numpy.ndarray validation functions for score data.

Sections:
    - Item-level score validation
    - Batch-level score validation
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np


# Item-level anomaly map validation
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


# Batch-level anomaly map validation


def validate_batch_pred_score(pred_score: np.ndarray | None, batch_size: int) -> np.ndarray | None:
    """Validate and convert the input batch of numpy prediction scores.

    Args:
        pred_score: The input prediction scores. Can be a numpy array or None.
        batch_size: The expected batch size.

    Returns:
        A numpy array of validated prediction scores, or None if the input was None.

    Raises:
        TypeError: If the input is not a numpy array.
        ValueError: If the input shape is invalid.

    Examples:
        >>> import numpy as np
        >>> pred_score = np.random.rand(4)
        >>> result = validate_batch_pred_score(pred_score, 4)
        >>> result.shape
        (4,)

        >>> validate_batch_pred_score(None, 4)
        None

        >>> validate_batch_pred_score(np.random.rand(4, 1), 4)
        Traceback (most recent call last):
            ...
        ValueError: Prediction score must be a 1-dimensional vector, got shape (4, 1).
    """
    if pred_score is None:
        return None
    if not isinstance(pred_score, np.ndarray):
        msg = f"Prediction score must be a np.ndarray, got {type(pred_score)}."
        raise TypeError(msg)
    pred_score = np.squeeze(pred_score)
    if pred_score.ndim != 1:
        msg = f"Prediction score must be a 1-dimensional vector, got shape {pred_score.shape}."
        raise ValueError(msg)
    if len(pred_score) != batch_size:
        msg = f"Prediction score must have length {batch_size}, got length {len(pred_score)}."
        raise ValueError(msg)
    return pred_score.astype(np.float32)


def validate_batch_anomaly_map(anomaly_map: np.ndarray | None, batch_size: int) -> np.ndarray | None:
    """Validate and convert the input batch of numpy anomaly maps.

    Args:
        anomaly_map: The input anomaly maps. Can be a numpy array or None.
        batch_size: The expected batch size.

    Returns:
        A numpy array of validated anomaly maps, or None if the input was None.

    Raises:
        TypeError: If the input is not a numpy array.
        ValueError: If the input shape is invalid.

    Examples:
        >>> import numpy as np
        >>> anomaly_map = np.random.rand(4, 224, 224)
        >>> result = validate_batch_anomaly_map(anomaly_map, 4)
        >>> result.shape
        (4, 224, 224)

        >>> validate_batch_anomaly_map(None, 4)
        None

        >>> validate_batch_anomaly_map(np.random.rand(4, 3, 224, 224), 4)
        Traceback (most recent call last):
            ...
        ValueError: Anomaly map must have 1 channel, got 3.
    """
    if anomaly_map is None:
        return None
    if not isinstance(anomaly_map, np.ndarray):
        msg = f"Anomaly map must be a np.ndarray, got {type(anomaly_map)}."
        raise TypeError(msg)
    if anomaly_map.ndim not in {2, 3, 4}:
        msg = f"Anomaly map must have shape [H, W] or [N, H, W] or [N, 1, H, W], got shape {anomaly_map.shape}."
        raise ValueError(msg)
    if anomaly_map.ndim == 2:
        if batch_size != 1:
            msg = f"Invalid shape for anomaly_map. Got shape {anomaly_map.shape} for batch size {batch_size}."
            raise ValueError(msg)
        anomaly_map = anomaly_map[np.newaxis, ...]
    if anomaly_map.ndim == 4:
        if anomaly_map.shape[1] != 1:
            msg = f"Anomaly map must have 1 channel, got {anomaly_map.shape[1]}."
            raise ValueError(msg)
        anomaly_map = np.squeeze(anomaly_map, axis=1)
    return anomaly_map.astype(np.float32)
