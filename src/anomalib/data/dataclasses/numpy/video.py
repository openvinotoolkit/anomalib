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

    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        return NumpyVideoValidator.validate_image(image)

    def _validate_gt_label(self, gt_label: np.ndarray | None) -> np.ndarray | None:
        return NumpyVideoValidator.validate_gt_label(gt_label)

    def _validate_gt_mask(self, gt_mask: np.ndarray | None) -> np.ndarray | None:
        return NumpyVideoValidator.validate_gt_mask(gt_mask)

    def _validate_mask_path(self, mask_path: str | None) -> str | None:
        return NumpyVideoValidator.validate_mask_path(mask_path)

    def _validate_anomaly_map(self, anomaly_map: np.ndarray | None) -> np.ndarray | None:
        return NumpyVideoValidator.validate_anomaly_map(anomaly_map)

    def _validate_pred_score(self, pred_score: np.ndarray | None) -> np.ndarray | None:
        return NumpyVideoValidator.validate_pred_score(pred_score)

    def _validate_pred_mask(self, pred_mask: np.ndarray | None) -> np.ndarray | None:
        return NumpyVideoValidator.validate_pred_mask(pred_mask)

    def _validate_pred_label(self, pred_label: np.ndarray | None) -> np.ndarray | None:
        return NumpyVideoValidator.validate_pred_label(pred_label)

    def _validate_video_path(self, video_path: str | None) -> str | None:
        return NumpyVideoValidator.validate_video_path(video_path)

    def _validate_original_image(self, original_image: np.ndarray | None) -> np.ndarray | None:
        return NumpyVideoValidator.validate_original_image(original_image)

    def _validate_target_frame(self, target_frame: int | None) -> int | None:
        return NumpyVideoValidator.validate_target_frame(target_frame)


@dataclass
class NumpyVideoBatch(
    BatchIterateMixin[NumpyVideoItem],
    _VideoInputFields[np.ndarray, np.ndarray, np.ndarray, list[str]],
    NumpyBatch,
):
    """Dataclass for a batch of video items in Anomalib datasets using numpy arrays.

    This class combines BatchIterateMixin, _VideoInputFields, and NumpyBatch for batches
    of video data. It supports batch operations and iteration over individual NumpyVideoItems.
    It ensures proper formatting for Anomalib's video-based models.
    """

    item_class = NumpyVideoItem

    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        return NumpyVideoBatchValidator.validate_image(image)

    def _validate_gt_label(self, gt_label: np.ndarray | None) -> np.ndarray | None:
        return NumpyVideoBatchValidator.validate_gt_label(gt_label, self.batch_size)

    def _validate_gt_mask(self, gt_mask: np.ndarray | None) -> np.ndarray | None:
        return NumpyVideoBatchValidator.validate_gt_mask(gt_mask, self.batch_size)

    def _validate_mask_path(self, mask_path: list[str] | None) -> list[str] | None:
        return NumpyVideoBatchValidator.validate_mask_path(mask_path, self.batch_size)

    def _validate_anomaly_map(self, anomaly_map: np.ndarray | None) -> np.ndarray | None:
        return NumpyVideoBatchValidator.validate_anomaly_map(anomaly_map, self.batch_size)

    def _validate_pred_score(self, pred_score: np.ndarray | None) -> np.ndarray | None:
        return NumpyVideoBatchValidator.validate_pred_score(pred_score)

    def _validate_pred_mask(self, pred_mask: np.ndarray | None) -> np.ndarray | None:
        return NumpyVideoBatchValidator.validate_pred_mask(pred_mask, self.batch_size)

    def _validate_pred_label(self, pred_label: np.ndarray | None) -> np.ndarray | None:
        return NumpyVideoBatchValidator.validate_pred_label(pred_label)

    def _validate_video_path(self, video_path: list[str] | None) -> list[str] | None:
        return NumpyVideoBatchValidator.validate_video_path(video_path)

    def _validate_original_image(self, original_image: np.ndarray | None) -> np.ndarray | None:
        return NumpyVideoBatchValidator.validate_original_image(original_image)

    def _validate_target_frame(self, target_frame: np.ndarray | None) -> np.ndarray | None:
        return NumpyVideoBatchValidator.validate_target_frame(target_frame)
