# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Validate GRD-Net region-of-interest data stored as PyTorch tensors."""

from collections.abc import Sequence
from pathlib import Path

from torchvision.tv_tensors import Mask

import torch
from anomalib.data.validators.path import validate_batch_path, validate_path


class GRDNetItemValidator:
    """Validate GRD-Net ROI fields for a single image item."""

    @staticmethod
    def validate_roi_mask(roi_mask: torch.Tensor | None) -> Mask | None:
        """Validate a single ROI mask.

        Args:
            roi_mask: ROI mask with shape ``[H, W]`` or ``[1, H, W]``.

        Returns:
            ROI mask converted to a boolean :class:`~torchvision.tv_tensors.Mask`, or ``None``.

        Raises:
            TypeError: If ``roi_mask`` is not a tensor.
            ValueError: If ``roi_mask`` is not a single-channel mask.
        """
        if roi_mask is None:
            return None
        if not isinstance(roi_mask, torch.Tensor):
            msg = f"ROI mask must be a torch.Tensor, got {type(roi_mask)}."
            raise TypeError(msg)
        if roi_mask.ndim not in {2, 3}:
            msg = f"ROI mask must have shape [H, W] or [1, H, W], got shape {roi_mask.shape}."
            raise ValueError(msg)
        if roi_mask.ndim == 3:
            if roi_mask.shape[0] != 1:
                msg = f"ROI mask must have 1 channel, got {roi_mask.shape[0]}."
                raise ValueError(msg)
            roi_mask = roi_mask.squeeze(0)
        return Mask(roi_mask, dtype=torch.bool)

    @staticmethod
    def validate_roi_mask_path(roi_mask_path: str | Path | None) -> str | None:
        """Validate a single ROI mask path.

        Empty strings are retained to represent generated full-image ROI masks.

        Args:
            roi_mask_path: Path to an ROI mask, an empty fallback path, or ``None``.

        Returns:
            Validated path as a string, or ``None``.
        """
        if roi_mask_path is None:
            return None
        if roi_mask_path == "":
            return ""
        return validate_path(roi_mask_path)


class GRDNetBatchValidator:
    """Validate GRD-Net ROI fields for an image batch."""

    @staticmethod
    def validate_roi_mask(roi_mask: torch.Tensor | None) -> Mask | None:
        """Validate a batch of ROI masks.

        Args:
            roi_mask: ROI masks with shape ``[B, H, W]`` or ``[B, 1, H, W]``.

        Returns:
            ROI masks converted to a boolean :class:`~torchvision.tv_tensors.Mask`, or ``None``.

        Raises:
            TypeError: If ``roi_mask`` is not a tensor.
            ValueError: If ``roi_mask`` does not contain single-channel batched masks.
        """
        if roi_mask is None:
            return None
        if not isinstance(roi_mask, torch.Tensor):
            msg = f"ROI mask batch must be a torch.Tensor, got {type(roi_mask)}."
            raise TypeError(msg)
        if roi_mask.ndim not in {3, 4}:
            msg = f"ROI mask batch must have shape [B, H, W] or [B, 1, H, W], got shape {roi_mask.shape}."
            raise ValueError(msg)
        if roi_mask.ndim == 4:
            if roi_mask.shape[1] != 1:
                msg = f"ROI mask batch must have 1 channel, got {roi_mask.shape[1]}."
                raise ValueError(msg)
            roi_mask = roi_mask.squeeze(1)
        return Mask(roi_mask, dtype=torch.bool)

    @staticmethod
    def validate_roi_mask_path(roi_mask_path: Sequence[str | Path] | None) -> list[str] | None:
        """Validate ROI mask paths for a batch.

        Args:
            roi_mask_path: Sequence of ROI mask paths, including empty fallback paths, or ``None``.

        Returns:
            Validated paths as strings, or ``None``.
        """
        if isinstance(roi_mask_path, str):
            msg = "ROI mask paths must be a sequence of paths, not a single string."
            raise TypeError(msg)
        return validate_batch_path(roi_mask_path)
