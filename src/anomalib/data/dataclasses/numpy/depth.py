"""Numpy-based depth dataclasses for Anomalib."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np

from anomalib.data.dataclasses.generic import BatchIterateMixin, _DepthInputFields
from anomalib.data.dataclasses.numpy.base import NumpyBatch, NumpyItem


@dataclass
class NumpyDepthItem(_DepthInputFields[np.ndarray, str], NumpyItem):
    """Dataclass for a single depth item in Anomalib datasets using numpy arrays.

    This class combines _DepthInputFields and NumpyItem for depth-based anomaly detection.
    It includes depth-specific fields and validation methods to ensure proper formatting
    for Anomalib's depth-based models.
    """

    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        return image

    def _validate_gt_label(self, gt_label: np.ndarray) -> np.ndarray:
        return gt_label

    def _validate_gt_mask(self, gt_mask: np.ndarray) -> np.ndarray:
        return gt_mask

    def _validate_mask_path(self, mask_path: str) -> str:
        return mask_path

    def _validate_anomaly_map(self, anomaly_map: np.ndarray) -> np.ndarray:
        return anomaly_map

    def _validate_pred_score(self, pred_score: np.ndarray) -> np.ndarray:
        return pred_score

    def _validate_pred_mask(self, pred_mask: np.ndarray) -> np.ndarray:
        return pred_mask

    def _validate_pred_label(self, pred_label: np.ndarray) -> np.ndarray:
        return pred_label

    def _validate_image_path(self, image_path: str) -> str:
        return image_path

    def _validate_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        return depth_map

    def _validate_depth_path(self, depth_path: str) -> str:
        return depth_path


class NumpyDepthBatch(
    BatchIterateMixin[NumpyDepthItem],
    _DepthInputFields[np.ndarray, list[str]],
    NumpyBatch,
):
    """Dataclass for a batch of depth items in Anomalib datasets using numpy arrays."""

    item_class = NumpyDepthItem

    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        return image

    def _validate_gt_label(self, gt_label: np.ndarray) -> np.ndarray:
        return gt_label

    def _validate_gt_mask(self, gt_mask: np.ndarray) -> np.ndarray:
        return gt_mask

    def _validate_mask_path(self, mask_path: list[str]) -> list[str]:
        return mask_path

    def _validate_anomaly_map(self, anomaly_map: np.ndarray) -> np.ndarray:
        return anomaly_map

    def _validate_pred_score(self, pred_score: np.ndarray) -> np.ndarray:
        return pred_score

    def _validate_pred_mask(self, pred_mask: np.ndarray) -> np.ndarray:
        return pred_mask

    def _validate_pred_label(self, pred_label: np.ndarray) -> np.ndarray:
        return pred_label

    def _validate_image_path(self, image_path: list[str]) -> list[str]:
        return image_path

    def _validate_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        return depth_map

    def _validate_depth_path(self, depth_path: list[str]) -> list[str]:
        return depth_path
