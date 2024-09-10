"""Numpy-based video dataclasses for Anomalib."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np

from anomalib.data.generic import BatchIterateMixin, _VideoInputFields
from anomalib.data.numpy.base import NumpyBatch, NumpyItem


@dataclass
class NumpyVideoItem(_VideoInputFields[np.ndarray, np.ndarray, np.ndarray, str], NumpyItem):
    """Dataclass for a single video item in Anomalib datasets using numpy arrays.

    This class combines _VideoInputFields and NumpyItem for video-based anomaly detection.
    It includes video-specific fields and validation methods to ensure proper formatting
    for Anomalib's video-based models.
    """

    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        return image

    def _validate_gt_label(self, gt_label: np.ndarray) -> np.ndarray:
        return gt_label

    def _validate_gt_mask(self, gt_mask: np.ndarray) -> np.ndarray:
        return gt_mask

    def _validate_mask_path(self, mask_path: str) -> str:
        return mask_path


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
        return image

    def _validate_gt_label(self, gt_label: np.ndarray) -> np.ndarray:
        return gt_label

    def _validate_gt_mask(self, gt_mask: np.ndarray) -> np.ndarray:
        return gt_mask

    def _validate_mask_path(self, mask_path: list[str]) -> list[str]:
        return mask_path

    def _validate_anomaly_map(self, anomaly_map: np.ndarray) -> np.ndarray:
        return anomaly_map
