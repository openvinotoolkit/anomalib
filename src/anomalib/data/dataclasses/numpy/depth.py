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

    @staticmethod
    def validate_image(image: np.ndarray) -> np.ndarray:
        """Validate the image."""
        return NumpyDepthValidator.validate_image(image)

    @staticmethod
    def validate_gt_label(gt_label: np.ndarray | None) -> np.ndarray | None:
        """Validate the ground truth label."""
        return NumpyDepthValidator.validate_gt_label(gt_label)

    @staticmethod
    def validate_gt_mask(gt_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the ground truth mask."""
        return NumpyDepthValidator.validate_gt_mask(gt_mask)

    @staticmethod
    def validate_mask_path(mask_path: str | None) -> str | None:
        """Validate the mask path."""
        return NumpyDepthValidator.validate_mask_path(mask_path)

    @staticmethod
    def validate_anomaly_map(anomaly_map: np.ndarray | None) -> np.ndarray | None:
        """Validate the anomaly map."""
        return NumpyDepthValidator.validate_anomaly_map(anomaly_map)

    @staticmethod
    def validate_pred_score(pred_score: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction score."""
        return NumpyDepthValidator.validate_pred_score(pred_score)

    @staticmethod
    def validate_pred_mask(pred_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction mask."""
        return NumpyDepthValidator.validate_pred_mask(pred_mask)

    @staticmethod
    def validate_pred_label(pred_label: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction label."""
        return NumpyDepthValidator.validate_pred_label(pred_label)

    @staticmethod
    def validate_image_path(image_path: str | None) -> str | None:
        """Validate the image path."""
        return NumpyDepthValidator.validate_image_path(image_path)

    @staticmethod
    def validate_depth_map(depth_map: np.ndarray | None) -> np.ndarray | None:
        """Validate the depth map."""
        return NumpyDepthValidator.validate_depth_map(depth_map)

    @staticmethod
    def validate_depth_path(depth_path: str | None) -> str | None:
        """Validate the depth path."""
        return NumpyDepthValidator.validate_depth_path(depth_path)


class NumpyDepthBatch(
    BatchIterateMixin[NumpyDepthItem],
    _DepthInputFields[np.ndarray, list[str]],
    NumpyBatch,
):
    """Dataclass for a batch of depth items in Anomalib datasets using numpy arrays."""

    item_class = NumpyDepthItem

    @staticmethod
    def validate_image(image: np.ndarray) -> np.ndarray:
        """Validate the image."""
        return NumpyDepthBatchValidator.validate_image(image)

    def validate_gt_label(self, gt_label: np.ndarray | None) -> np.ndarray | None:
        """Validate the ground truth label."""
        return NumpyDepthBatchValidator.validate_gt_label(gt_label, self.batch_size)

    def validate_gt_mask(self, gt_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the ground truth mask."""
        return NumpyDepthBatchValidator.validate_gt_mask(gt_mask, self.batch_size)

    def validate_mask_path(self, mask_path: list[str] | None) -> list[str] | None:
        """Validate the mask path."""
        return NumpyDepthBatchValidator.validate_mask_path(mask_path, self.batch_size)

    def validate_anomaly_map(self, anomaly_map: np.ndarray | None) -> np.ndarray | None:
        """Validate the anomaly map."""
        return NumpyDepthBatchValidator.validate_anomaly_map(anomaly_map, self.batch_size)

    @staticmethod
    def validate_pred_score(pred_score: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction score."""
        return NumpyDepthBatchValidator.validate_pred_score(pred_score)

    def validate_pred_mask(self, pred_mask: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction mask."""
        return NumpyDepthBatchValidator.validate_pred_mask(pred_mask, self.batch_size)

    @staticmethod
    def validate_pred_label(pred_label: np.ndarray | None) -> np.ndarray | None:
        """Validate the prediction label."""
        return NumpyDepthBatchValidator.validate_pred_label(pred_label)

    @staticmethod
    def validate_image_path(image_path: list[str] | None) -> list[str] | None:
        """Validate the image path."""
        return NumpyDepthBatchValidator.validate_image_path(image_path)

    def validate_depth_map(self, depth_map: np.ndarray | None) -> np.ndarray | None:
        """Validate the depth map."""
        return NumpyDepthBatchValidator.validate_depth_map(depth_map, self.batch_size)

    @staticmethod
    def validate_depth_path(depth_path: list[str] | None) -> list[str] | None:
        """Validate the depth path."""
        return NumpyDepthBatchValidator.validate_depth_path(depth_path)
