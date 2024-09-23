"""Numpy-based video dataclasses for Anomalib."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np

from anomalib.data.dataclasses.generic import BatchIterateMixin, _VideoInputFields
from anomalib.data.dataclasses.numpy.base import NumpyBatch, NumpyItem
from anomalib.data.validators.numpy.video import NumpyVideoBatchValidator, NumpyVideoValidator


@dataclass
class NumpyVideoItem(_VideoInputFields[np.ndarray, np.ndarray, np.ndarray, str], NumpyItem):
    """Dataclass for a single video item in Anomalib datasets using numpy arrays.

    This class combines _VideoInputFields and NumpyItem for video-based anomaly detection.
    It includes video-specific fields and validation methods to ensure proper formatting
    for Anomalib's video-based models.
    """

    @staticmethod
    def validate_image(image: np.ndarray) -> np.ndarray:
        """Validate the image."""
        return NumpyVideoValidator.validate_image(image)

    @staticmethod
    def validate_gt_label(gt_label: np.ndarray | None) -> np.ndarray | None:
        """Validate the ground truth label."""
        return NumpyVideoValidator.validate_gt_label(gt_label)

    @staticmethod
    def validate_gt_mask(gt_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the ground truth mask."""
        return NumpyVideoValidator.validate_gt_mask(gt_mask)

    @staticmethod
    def validate_mask_path(mask_path: str | None) -> str | None:
        """Validate the mask path."""
        return NumpyVideoValidator.validate_mask_path(mask_path)

    @staticmethod
    def validate_anomaly_map(anomaly_map: np.ndarray | None) -> np.ndarray | None:
        """Validate the anomaly map."""
        return NumpyVideoValidator.validate_anomaly_map(anomaly_map)

    @staticmethod
    def validate_pred_score(pred_score: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction score."""
        return NumpyVideoValidator.validate_pred_score(pred_score)

    @staticmethod
    def validate_pred_mask(pred_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction mask."""
        return NumpyVideoValidator.validate_pred_mask(pred_mask)

    @staticmethod
    def validate_pred_label(pred_label: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction label."""
        return NumpyVideoValidator.validate_pred_label(pred_label)

    @staticmethod
    def validate_video_path(video_path: str | None) -> str | None:
        """Validate the video path."""
        return NumpyVideoValidator.validate_video_path(video_path)

    @staticmethod
    def validate_original_image(original_image: np.ndarray | None) -> np.ndarray | None:
        """Validate the original image."""
        return NumpyVideoValidator.validate_original_image(original_image)

    @staticmethod
    def validate_target_frame(target_frame: int | None) -> int | None:
        """Validate the target frame."""
        return NumpyVideoValidator.validate_target_frame(target_frame)


@dataclass
class NumpyVideoBatch(
    BatchIterateMixin[NumpyVideoItem],
    NumpyVideoBatchValidator,
    _VideoInputFields[np.ndarray, np.ndarray, np.ndarray, list[str]],
    NumpyBatch,
):
    """Dataclass for a batch of video items in Anomalib datasets using numpy arrays.

    This class combines BatchIterateMixin, _VideoInputFields, and NumpyBatch for batches
    of video data. It supports batch operations and iteration over individual NumpyVideoItems.
    It ensures proper formatting for Anomalib's video-based models.
    """

    item_class = NumpyVideoItem
