"""Validate torch label data."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch


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
