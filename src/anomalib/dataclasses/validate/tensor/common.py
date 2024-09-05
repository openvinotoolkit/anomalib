"""Common validation functions."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision.tv_tensors import Mask


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


def validate_label(label: int | torch.Tensor) -> torch.Tensor:
    """Validate and convert the input label to a boolean PyTorch tensor.

    Args:
        label: The input label to validate. Must be either an integer or a scalar PyTorch tensor.

    Returns:
        The validated label as a boolean PyTorch tensor.

    Raises:
        TypeError: If the input is not an int or PyTorch tensor.
        ValueError: If the tensor is not a scalar or is a floating point type.

    Examples:
        >>> import torch
        >>> from anomalib.data.io.validate import validate_torch_label

        >>> # Integer input
        >>> validate_torch_label(1)
        tensor(True)

        >>> validate_torch_label(0)
        tensor(False)

        >>> # PyTorch tensor input
        >>> validate_torch_label(torch.tensor(1))
        tensor(True)

        >>> validate_torch_label(torch.tensor(0))
        tensor(False)

        >>> # Invalid inputs
        >>> validate_torch_label(torch.tensor([1, 0]))
        Traceback (most recent call last):
            ...
        ValueError: Ground truth label must be a scalar, got shape torch.Size([2]).

        >>> validate_torch_label(torch.tensor(1.0))
        Traceback (most recent call last):
            ...
        ValueError: Ground truth label must be boolean or integer, got torch.float32.

        >>> validate_torch_label("invalid")
        Traceback (most recent call last):
            ...
        TypeError: Ground truth label must be an integer or a torch.Tensor, got <class 'str'>.
    """
    if isinstance(label, int):
        return torch.tensor(label, dtype=torch.bool)
    if isinstance(label, torch.Tensor):
        if label.ndim != 0:
            msg = f"Ground truth label must be a scalar, got shape {label.shape}."
            raise ValueError(msg)
        if torch.is_floating_point(label):
            msg = f"Ground truth label must be boolean or integer, got {label.dtype}."
            raise ValueError(msg)
        return label.bool()
    msg = f"Ground truth label must be an integer or a torch.Tensor, got {type(label)}."
    raise TypeError(msg)


def validate_mask(mask: torch.Tensor) -> Mask:
    """Validate and convert the input PyTorch mask.

    Args:
        mask: The input mask to validate. Must be a PyTorch tensor.

    Returns:
        The validated mask as a torchvision Mask.

    Raises:
        TypeError: If the input is not a PyTorch tensor.
        ValueError: If the mask dimensions are invalid.

    Examples:
        >>> import torch
        >>> from anomalib.data.io.validate import validate_torch_mask

        >>> # 2D input
        >>> torch_mask = torch.randint(0, 2, (100, 100))
        >>> result = validate_torch_mask(torch_mask)
        >>> isinstance(result, Mask)
        True
        >>> result.shape
        torch.Size([100, 100])

        >>> # 3D input (will be squeezed)
        >>> torch_mask_3d = torch.randint(0, 2, (1, 100, 100))
        >>> result = validate_torch_mask(torch_mask_3d)
        >>> result.shape
        torch.Size([100, 100])

        >>> # Invalid input
        >>> validate_torch_mask(torch.rand(3, 100, 100))
        Traceback (most recent call last):
            ...
        ValueError: Ground truth mask must have shape [H, W] or [1, H, W], got shape torch.Size([3, 100, 100]).

        >>> validate_torch_mask(np.random.randint(0, 2, (100, 100)))
        Traceback (most recent call last):
            ...
        TypeError: Ground truth mask must be a torch.Tensor, got <class 'numpy.ndarray'>.
    """
    if not isinstance(mask, torch.Tensor):
        msg = f"Ground truth mask must be a torch.Tensor, got {type(mask)}."
        raise TypeError(msg)

    if mask.ndim not in {2, 3}:
        msg = f"Ground truth mask must have shape [H, W] or [1, H, W], got shape {mask.shape}."
        raise ValueError(msg)
    if mask.ndim == 3:
        if mask.shape[0] != 1:
            msg = f"Ground truth mask must have 1 channel, got {mask.shape[0]}."
            raise ValueError(msg)
        mask = mask.squeeze(0)
    return Mask(mask, dtype=torch.bool)


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


def validate_pred_mask(pred_mask: torch.Tensor) -> Mask:
    """Validate and convert the input PyTorch predicted mask.

    Args:
        pred_mask: The input predicted mask to validate. Must be a PyTorch tensor.

    Returns:
        The validated predicted mask as a torchvision Mask.

    Raises:
        TypeError: If the input is not a PyTorch tensor.
        ValueError: If the mask dimensions are invalid.

    Examples:
        >>> import torch
        >>> from anomalib.data.io.validate import validate_torch_pred_mask

        >>> # 2D input
        >>> torch_mask = torch.randint(0, 2, (100, 100))
        >>> result = validate_torch_pred_mask(torch_mask)
        >>> isinstance(result, Mask)
        True
        >>> result.shape
        torch.Size([100, 100])

        >>> # 3D input (will be squeezed)
        >>> torch_mask_3d = torch.randint(0, 2, (1, 100, 100))
        >>> result = validate_torch_pred_mask(torch_mask_3d)
        >>> result.shape
        torch.Size([100, 100])

        >>> # Invalid input
        >>> validate_torch_pred_mask(torch.rand(3, 100, 100))
        Traceback (most recent call last):
            ...
        ValueError: Predicted mask must have 1 channel, got 3.
    """
    if not isinstance(pred_mask, torch.Tensor):
        msg = f"Predicted mask must be a torch.Tensor, got {type(pred_mask)}."
        raise TypeError(msg)

    if pred_mask.ndim not in {2, 3}:
        msg = f"Predicted mask must have shape [H, W] or [1, H, W], got shape {pred_mask.shape}."
        raise ValueError(msg)
    if pred_mask.ndim == 3:
        if pred_mask.shape[0] != 1:
            msg = f"Predicted mask must have 1 channel, got {pred_mask.shape[0]}."
            raise ValueError(msg)
        pred_mask = pred_mask.squeeze(0)
    return Mask(pred_mask, dtype=torch.bool)


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


def validate_pred_label(pred_label: torch.Tensor | int) -> torch.Tensor:
    """Validate and convert the input PyTorch predicted label.

    Args:
        pred_label: The input predicted label to validate. Can be a PyTorch tensor or int.

    Returns:
        The validated predicted label as a PyTorch tensor.

    Raises:
        TypeError: If the input is not a PyTorch tensor or int.
        ValueError: If the predicted label is not a scalar.

    Examples:
        >>> import torch
        >>> from anomalib.data.io.validate import validate_torch_pred_label

        >>> # Scalar tensor input
        >>> torch_label = torch.tensor(1)
        >>> result = validate_torch_pred_label(torch_label)
        >>> isinstance(result, torch.Tensor) and result.dtype == torch.bool
        True
        >>> result.item()
        True

        >>> # Integer input
        >>> result = validate_torch_pred_label(0)
        >>> isinstance(result, torch.Tensor) and result.dtype == torch.bool
        True
        >>> result.item()
        False

        >>> # Invalid input
        >>> validate_torch_pred_label(torch.tensor([0, 1]))
        Traceback (most recent call last):
            ...
        ValueError: Predicted label must be a scalar, got shape torch.Size([2]).
    """
    if isinstance(pred_label, int):
        pred_label = torch.tensor(pred_label)
    if not isinstance(pred_label, torch.Tensor):
        msg = f"Predicted label must be a torch.Tensor or int, got {type(pred_label)}."
        raise TypeError(msg)
    pred_label = pred_label.squeeze()
    if pred_label.ndim != 0:
        msg = f"Predicted label must be a scalar, got shape {pred_label.shape}."
        raise ValueError(msg)
    return pred_label.to(torch.bool)
