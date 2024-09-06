"""Validate torch score data."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision.tv_tensors import Mask


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
