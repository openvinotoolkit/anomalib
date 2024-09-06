"""Validate torch mask data."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision.tv_tensors import Mask


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
