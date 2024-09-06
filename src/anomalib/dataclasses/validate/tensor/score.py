"""Validate torch score data.

Sections:
    - Item-level score validation
    - Batch score validation
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from torchvision.tv_tensors import Mask


# Item-level score validation
def validate_pred_score(pred_score: torch.Tensor | float) -> torch.Tensor:
    """Validate and convert the input PyTorch prediction score.

    Args:
        pred_score: The input prediction score to validate. Can be a PyTorch tensor or float.

    Returns:
        The validated prediction score as a PyTorch tensor.

    Raises:
        TypeError: If the input is not a PyTorch tensor or float.
        ValueError: If the prediction score is not a scalar.

    Examples:
        >>> import torch
        >>> from anomalib.data.io.validate import validate_torch_pred_score

        >>> # Scalar tensor input
        >>> torch_score = torch.tensor(0.8)
        >>> result = validate_torch_pred_score(torch_score)
        >>> isinstance(result, torch.Tensor) and result.dtype == torch.float32
        True
        >>> result.item()
        0.8

        >>> # Float input
        >>> result = validate_torch_pred_score(0.7)
        >>> isinstance(result, torch.Tensor) and result.dtype == torch.float32
        True
        >>> result.item()
        0.7

        >>> # Invalid input
        >>> validate_torch_pred_score(torch.rand(2))
        Traceback (most recent call last):
            ...
        ValueError: Prediction score must be a scalar, got shape torch.Size([2]).
    """
    if isinstance(pred_score, float):
        pred_score = torch.tensor(pred_score)
    if not isinstance(pred_score, torch.Tensor):
        msg = f"Prediction score must be a torch.Tensor or float, got {type(pred_score)}."
        raise TypeError(msg)
    pred_score = pred_score.squeeze()
    if pred_score.ndim != 0:
        msg = f"Prediction score must be a scalar, got shape {pred_score.shape}."
        raise ValueError(msg)
    return pred_score.to(torch.float32)


def validate_anomaly_map(anomaly_map: torch.Tensor) -> Mask:
    """Validate and convert the input PyTorch anomaly map.

    Args:
        anomaly_map: The input anomaly map to validate. Must be a PyTorch tensor.

    Returns:
        The validated anomaly map as a torchvision Mask.

    Raises:
        TypeError: If the input is not a PyTorch tensor.
        ValueError: If the anomaly map dimensions are invalid.

    Examples:
        >>> import torch
        >>> from anomalib.data.io.validate import validate_torch_anomaly_map

        >>> # 2D input
        >>> torch_map = torch.rand(100, 100)
        >>> result = validate_torch_anomaly_map(torch_map)
        >>> isinstance(result, Mask)
        True
        >>> result.shape
        torch.Size([100, 100])

        >>> # 3D input (will be squeezed)
        >>> torch_map_3d = torch.rand(1, 100, 100)
        >>> result = validate_torch_anomaly_map(torch_map_3d)
        >>> result.shape
        torch.Size([100, 100])

        >>> # Invalid input
        >>> validate_torch_anomaly_map(torch.rand(3, 100, 100))
        Traceback (most recent call last):
            ...
        ValueError: Anomaly map must have 1 channel, got 3.
    """
    if not isinstance(anomaly_map, torch.Tensor):
        msg = f"Anomaly map must be a torch.Tensor, got {type(anomaly_map)}."
        raise TypeError(msg)

    if anomaly_map.ndim not in {2, 3}:
        msg = f"Anomaly map must have shape [H, W] or [1, H, W], got shape {anomaly_map.shape}."
        raise ValueError(msg)
    if anomaly_map.ndim == 3:
        if anomaly_map.shape[0] != 1:
            msg = f"Anomaly map must have 1 channel, got {anomaly_map.shape[0]}."
            raise ValueError(msg)
        anomaly_map = anomaly_map.squeeze(0)
    return Mask(anomaly_map, dtype=torch.float32)


# Batch score validation
def validate_batch_pred_score(pred_score: torch.Tensor | None, batch_size: int) -> torch.Tensor | None:
    """Validate and convert the input batch of PyTorch prediction scores.

    Args:
        pred_score: The input prediction scores. Can be a PyTorch tensor or None.
        batch_size: The expected batch size.

    Returns:
        A PyTorch tensor of validated prediction scores, or None if the input was None.

    Raises:
        TypeError: If the input is not a PyTorch tensor.
        ValueError: If the input shape is invalid.

    Examples:
        >>> import torch
        >>> pred_score = torch.rand(4)
        >>> result = validate_batch_pred_score(pred_score, 4)
        >>> result.shape
        torch.Size([4])

        >>> validate_batch_pred_score(None, 4)
        None

        >>> validate_batch_pred_score(torch.rand(4, 1), 4)
        Traceback (most recent call last):
            ...
        ValueError: Prediction score must be a 1-dimensional vector, got shape torch.Size([4, 1]).
    """
    if pred_score is None:
        return None
    if not isinstance(pred_score, torch.Tensor):
        msg = f"Prediction score must be a torch.Tensor, got {type(pred_score)}."
        raise TypeError(msg)
    pred_score = pred_score.squeeze()
    if pred_score.ndim != 1:
        msg = f"Prediction score must be a 1-dimensional vector, got shape {pred_score.shape}."
        raise ValueError(msg)
    if len(pred_score) != batch_size:
        msg = f"Prediction score must have length {batch_size}, got length {len(pred_score)}."
        raise ValueError(msg)
    return pred_score.to(torch.float32)


def validate_batch_anomaly_map(anomaly_map: torch.Tensor | np.ndarray | None, batch_size: int) -> Mask | None:
    """Validate and convert the input batch of PyTorch anomaly maps.

    Args:
        anomaly_map: The input anomaly maps. Can be a PyTorch tensor or None.
        batch_size: The expected batch size.

    Returns:
        A torchvision Mask object of validated anomaly maps, or None if the input was None.

    Raises:
        TypeError: If the input is not a PyTorch tensor.
        ValueError: If the input shape is invalid.

    Examples:
        >>> import torch
        >>> from torchvision.tv_tensors import Mask
        >>> anomaly_map = torch.rand(4, 224, 224)
        >>> result = validate_batch_anomaly_map(anomaly_map, 4)
        >>> isinstance(result, Mask)
        True
        >>> result.shape
        torch.Size([4, 224, 224])

        >>> validate_batch_anomaly_map(None, 4)
        None

        >>> validate_batch_anomaly_map(torch.rand(4, 3, 224, 224), 4)
        Traceback (most recent call last):
            ...
        ValueError: Anomaly map must have 1 channel, got 3.
    """
    if anomaly_map is None:
        return None
    if not isinstance(anomaly_map, torch.Tensor):
        try:
            anomaly_map = torch.tensor(anomaly_map)
        except Exception as e:
            msg = "Anomaly map must be a torch.Tensor. Tried to convert to torch.Tensor but failed."
            raise ValueError(msg) from e
    if anomaly_map.ndim not in {2, 3, 4}:
        msg = f"Anomaly map must have shape [H, W] or [N, H, W] or [N, 1, H, W], got shape {anomaly_map.shape}."
        raise ValueError(msg)
    if anomaly_map.ndim == 2:
        if batch_size != 1:
            msg = f"Invalid shape for anomaly_map. Got shape {anomaly_map.shape} for batch size {batch_size}."
            raise ValueError(msg)
        anomaly_map = anomaly_map.unsqueeze(0)
    if anomaly_map.ndim == 4:
        if anomaly_map.shape[1] != 1:
            msg = f"Anomaly map must have 1 channel, got {anomaly_map.shape[1]}."
            raise ValueError(msg)
        anomaly_map = anomaly_map.squeeze(1)
    return Mask(anomaly_map, dtype=torch.float32)
