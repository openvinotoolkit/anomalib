"""Validate torch label data.

This module contains functions for validating and converting label data
in PyTorch tensors.

Sections:
- Item-level label validation
- Batch-level label validation
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import torch


# Item-level label validation
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


# Batch-level label validation
def validate_batch_label(gt_label: torch.Tensor | Sequence[int] | None, batch_size: int) -> torch.Tensor | None:
    """Validate and convert the input batch of labels to a boolean PyTorch tensor.

    Args:
        gt_label: The input ground truth labels. Can be a PyTorch tensor, a sequence of integers, or None.
        batch_size: The expected batch size.

    Returns:
        A boolean PyTorch tensor of validated labels, or None if the input was None.

    Raises:
        TypeError: If the input is not a PyTorch tensor or a sequence of integers.
        ValueError: If the input shape or type is invalid.

    Examples:
        >>> import torch
        >>> validate_batch_label(torch.tensor([0, 1, 1, 0]), 4)
        tensor([False,  True,  True, False])

        >>> validate_batch_label([0, 1, 1, 0], 4)
        tensor([False,  True,  True, False])

        >>> validate_batch_label(None, 4)
        None

        >>> validate_batch_label(torch.tensor([0.5, 1.5]), 2)
        Traceback (most recent call last):
            ...
        ValueError: Ground truth label must be boolean or integer, got torch.float32.
    """
    if gt_label is None:
        return None
    if isinstance(gt_label, Sequence):
        gt_label = torch.tensor(gt_label)
    if not isinstance(gt_label, torch.Tensor):
        msg = f"Ground truth label must be a sequence of integers or a torch.Tensor, got {type(gt_label)}."
        raise TypeError(msg)
    if gt_label.ndim != 1:
        msg = f"Ground truth label must be a 1-dimensional vector, got shape {gt_label.shape}."
        raise ValueError(msg)
    if len(gt_label) != batch_size:
        msg = f"Ground truth label must have length {batch_size}, got length {len(gt_label)}."
        raise ValueError(msg)
    if torch.is_floating_point(gt_label):
        msg = f"Ground truth label must be boolean or integer, got {gt_label.dtype}."
        raise ValueError(msg)
    return gt_label.bool()


def validate_batch_pred_label(pred_label: torch.Tensor | None, batch_size: int) -> torch.Tensor | None:
    """Validate and convert the input batch of PyTorch predicted labels.

    Args:
        pred_label: The input predicted labels. Can be a PyTorch tensor or None.
        batch_size: The expected batch size.

    Returns:
        A PyTorch tensor of validated predicted labels, or None if the input was None.

    Raises:
        TypeError: If the input is not a PyTorch tensor.
        ValueError: If the input shape is invalid.

    Examples:
        >>> import torch
        >>> pred_label = torch.randint(0, 2, (4,))
        >>> result = validate_batch_pred_label(pred_label, 4)
        >>> result.shape
        torch.Size([4])

        >>> validate_batch_pred_label(None, 4)
        None

        >>> validate_batch_pred_label(torch.rand(4, 1), 4)
        Traceback (most recent call last):
            ...
        ValueError: Predicted label must be a 1-dimensional vector, got shape torch.Size([4, 1]).
    """
    if pred_label is None:
        return None
    if not isinstance(pred_label, torch.Tensor):
        msg = f"Predicted label must be a torch.Tensor, got {type(pred_label)}."
        raise TypeError(msg)
    pred_label = pred_label.squeeze()
    if pred_label.ndim != 1:
        msg = f"Predicted label must be a 1-dimensional vector, got shape {pred_label.shape}."
        raise ValueError(msg)
    if len(pred_label) != batch_size:
        msg = f"Predicted label must have length {batch_size}, got length {len(pred_label)}."
        raise ValueError(msg)
    return pred_label.to(torch.bool)
