"""Validate numpy depth data."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import numpy as np

from anomalib.data.validators.numpy.image import NumpyImageBatchValidator, NumpyImageValidator
from anomalib.data.validators.path import validate_path


class NumpyDepthValidator:
    """Validate numpy.ndarray data for depth images."""

    @staticmethod
    def validate_image(image: np.ndarray) -> np.ndarray:
        """Validate the image array."""
        return NumpyImageValidator.validate_image(image)

    @staticmethod
    def validate_gt_label(label: int | np.ndarray | None) -> np.ndarray | None:
        """Validate the ground truth label."""
        return NumpyImageValidator.validate_gt_label(label)

    @staticmethod
    def validate_gt_mask(mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the ground truth mask."""
        return NumpyImageValidator.validate_gt_mask(mask)

    @staticmethod
    def validate_mask_path(mask_path: str | None) -> str | None:
        """Validate the mask path."""
        return NumpyImageValidator.validate_mask_path(mask_path)

    @staticmethod
    def validate_anomaly_map(anomaly_map: np.ndarray | None) -> np.ndarray | None:
        """Validate the anomaly map."""
        return NumpyImageValidator.validate_anomaly_map(anomaly_map)

    @staticmethod
    def validate_pred_score(
        pred_score: np.ndarray | float | None,
        anomaly_map: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Validate the prediction score."""
        return NumpyImageValidator.validate_pred_score(pred_score, anomaly_map)

    @staticmethod
    def validate_pred_mask(pred_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction mask."""
        return NumpyImageValidator.validate_pred_mask(pred_mask)

    @staticmethod
    def validate_pred_label(pred_label: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction label."""
        return NumpyImageValidator.validate_pred_label(pred_label)

    @staticmethod
    def validate_image_path(image_path: str | None) -> str | None:
        """Validate the image path."""
        return NumpyImageValidator.validate_image_path(image_path)

    @staticmethod
    def validate_depth_map(depth_map: np.ndarray | None) -> np.ndarray | None:
        """Validate the depth map."""
        if depth_map is None:
            return None
        if not isinstance(depth_map, np.ndarray):
            msg = f"Depth map must be a numpy array, got {type(depth_map)}."
            raise TypeError(msg)
        if depth_map.ndim not in {2, 3}:
            msg = f"Depth map must have shape [H, W] or [H, W, 1], got shape {depth_map.shape}."
            raise ValueError(msg)
        if depth_map.ndim == 3 and depth_map.shape[2] != 1:
            msg = f"Depth map with 3 dimensions must have 1 channel, got {depth_map.shape[2]}."
            raise ValueError(msg)
        return depth_map.astype(np.float32)

    @staticmethod
    def validate_depth_path(depth_path: str | None) -> str | None:
        """Validate the depth path."""
        return validate_path(depth_path) if depth_path else None


class NumpyDepthBatchValidator:
    """Validate numpy.ndarray data for batches of depth images."""

    @staticmethod
    def validate_image(image: np.ndarray) -> np.ndarray:
        """Validate the image batch array."""
        return NumpyImageBatchValidator.validate_image(image)

    @staticmethod
    def validate_gt_label(gt_label: np.ndarray | Sequence[int] | None, batch_size: int) -> np.ndarray | None:
        """Validate the ground truth label batch."""
        return NumpyImageBatchValidator.validate_gt_label(gt_label, batch_size)

    @staticmethod
    def validate_gt_mask(gt_mask: np.ndarray | None, batch_size: int) -> np.ndarray | None:
        """Validate the ground truth mask batch."""
        return NumpyImageBatchValidator.validate_gt_mask(gt_mask, batch_size)

    @staticmethod
    def validate_mask_path(mask_path: Sequence[str] | None, batch_size: int) -> list[str] | None:
        """Validate the mask paths for a batch."""
        return NumpyImageBatchValidator.validate_mask_path(mask_path, batch_size)

    @staticmethod
    def validate_anomaly_map(anomaly_map: np.ndarray | None, batch_size: int) -> np.ndarray | None:
        """Validate the anomaly map batch."""
        return NumpyImageBatchValidator.validate_anomaly_map(anomaly_map, batch_size)

    @staticmethod
    def validate_pred_score(pred_score: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction scores for a batch."""
        return NumpyImageBatchValidator.validate_pred_score(pred_score)

    @staticmethod
    def validate_pred_mask(pred_mask: np.ndarray | None, batch_size: int) -> np.ndarray | None:
        """Validate the prediction mask batch."""
        return NumpyImageBatchValidator.validate_pred_mask(pred_mask, batch_size)

    @staticmethod
    def validate_pred_label(pred_label: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction label batch."""
        return NumpyImageBatchValidator.validate_pred_label(pred_label)

    @staticmethod
    def validate_image_path(image_path: list[str] | None) -> list[str] | None:
        """Validate the image paths for a batch."""
        return NumpyImageBatchValidator.validate_image_path(image_path)

    @staticmethod
    def validate_depth_map(depth_map: np.ndarray | None, batch_size: int) -> np.ndarray | None:
        """Validate the depth map batch."""
        if depth_map is None:
            return None
        if not isinstance(depth_map, np.ndarray):
            msg = f"Depth map batch must be a numpy array, got {type(depth_map)}."
            raise TypeError(msg)
        if depth_map.ndim not in {3, 4}:
            msg = f"Depth map batch must have shape [N, H, W] or [N, H, W, 1], got shape {depth_map.shape}."
            raise ValueError(msg)
        if depth_map.shape[0] != batch_size:
            msg = f"Depth map batch size must be {batch_size}, got {depth_map.shape[0]}."
            raise ValueError(msg)
        if depth_map.ndim == 4 and depth_map.shape[3] != 1:
            msg = f"Depth map batch with 4 dimensions must have 1 channel, got {depth_map.shape[3]}."
            raise ValueError(msg)
        return depth_map.astype(np.float32)

    @staticmethod
    def validate_depth_path(depth_path: list[str] | None) -> list[str] | None:
        """Validate the depth paths for a batch."""
        if depth_path is None:
            return None
        if not isinstance(depth_path, list):
            msg = f"Depth path must be a list of strings, got {type(depth_path)}."
            raise TypeError(msg)
        return [validate_path(path) for path in depth_path]
