"""Numpy-based depth dataclasses for Anomalib."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np

from anomalib.data.dataclasses.generic import BatchIterateMixin, _DepthInputFields
from anomalib.data.dataclasses.numpy.base import NumpyBatch, NumpyItem
from anomalib.data.validators.numpy.depth import NumpyDepthBatchValidator, NumpyDepthValidator


@dataclass
class NumpyDepthItem(_DepthInputFields[np.ndarray, str], NumpyItem):
    """Dataclass for a single depth item in Anomalib datasets using numpy arrays.

    This class combines _DepthInputFields and NumpyItem for depth-based anomaly detection.
    It includes depth-specific fields and validation methods to ensure proper formatting
    for Anomalib's depth-based models.
    """

    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        return NumpyDepthValidator.validate_image(image)

    def _validate_gt_label(self, gt_label: np.ndarray | None) -> np.ndarray | None:
        return NumpyDepthValidator.validate_gt_label(gt_label)

    def _validate_gt_mask(self, gt_mask: np.ndarray | None) -> np.ndarray | None:
        return NumpyDepthValidator.validate_gt_mask(gt_mask)

    def _validate_mask_path(self, mask_path: str | None) -> str | None:
        return NumpyDepthValidator.validate_mask_path(mask_path)

    def _validate_anomaly_map(self, anomaly_map: np.ndarray | None) -> np.ndarray | None:
        return NumpyDepthValidator.validate_anomaly_map(anomaly_map)

    def _validate_pred_score(self, pred_score: np.ndarray | None) -> np.ndarray | None:
        return NumpyDepthValidator.validate_pred_score(pred_score)

    def _validate_pred_mask(self, pred_mask: np.ndarray | None) -> np.ndarray | None:
        return NumpyDepthValidator.validate_pred_mask(pred_mask)

    def _validate_pred_label(self, pred_label: np.ndarray | None) -> np.ndarray | None:
        return NumpyDepthValidator.validate_pred_label(pred_label)

    def _validate_image_path(self, image_path: str | None) -> str | None:
        return NumpyDepthValidator.validate_image_path(image_path)

    def _validate_depth_map(self, depth_map: np.ndarray | None) -> np.ndarray | None:
        return NumpyDepthValidator.validate_depth_map(depth_map)

    def _validate_depth_path(self, depth_path: str | None) -> str | None:
        return NumpyDepthValidator.validate_depth_path(depth_path)


class NumpyDepthBatch(
    BatchIterateMixin[NumpyDepthItem],
    _DepthInputFields[np.ndarray, list[str]],
    NumpyBatch,
):
    """Dataclass for a batch of depth items in Anomalib datasets using numpy arrays."""

    item_class = NumpyDepthItem

    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        return NumpyDepthBatchValidator.validate_image(image)

    def _validate_gt_label(self, gt_label: np.ndarray | None) -> np.ndarray | None:
        return NumpyDepthBatchValidator.validate_gt_label(gt_label, self.batch_size)

    def _validate_gt_mask(self, gt_mask: np.ndarray | None) -> np.ndarray | None:
        return NumpyDepthBatchValidator.validate_gt_mask(gt_mask, self.batch_size)

    def _validate_mask_path(self, mask_path: list[str] | None) -> list[str] | None:
        return NumpyDepthBatchValidator.validate_mask_path(mask_path, self.batch_size)

    def _validate_anomaly_map(self, anomaly_map: np.ndarray | None) -> np.ndarray | None:
        return NumpyDepthBatchValidator.validate_anomaly_map(anomaly_map, self.batch_size)

    def _validate_pred_score(self, pred_score: np.ndarray | None) -> np.ndarray | None:
        return NumpyDepthBatchValidator.validate_pred_score(pred_score)

    def _validate_pred_mask(self, pred_mask: np.ndarray | None) -> np.ndarray | None:
        return NumpyDepthBatchValidator.validate_pred_mask(pred_mask, self.batch_size)

    def _validate_pred_label(self, pred_label: np.ndarray | None) -> np.ndarray | None:
        return NumpyDepthBatchValidator.validate_pred_label(pred_label)

    def _validate_image_path(self, image_path: list[str] | None) -> list[str] | None:
        return NumpyDepthBatchValidator.validate_image_path(image_path)

    def _validate_depth_map(self, depth_map: np.ndarray | None) -> np.ndarray | None:
        return NumpyDepthBatchValidator.validate_depth_map(depth_map, self.batch_size)

    def _validate_depth_path(self, depth_path: list[str] | None) -> list[str] | None:
        return NumpyDepthBatchValidator.validate_depth_path(depth_path)
