"""Numpy-based video dataclasses for Anomalib."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np

from anomalib.data.dataclasses.generic import BatchIterateMixin, _VideoInputFields
from anomalib.data.dataclasses.numpy.base import NumpyBatch, NumpyItem


@dataclass
class NumpyVideoItem(_VideoInputFields[np.ndarray, np.ndarray, np.ndarray, str], NumpyItem):
    """Dataclass for a single video item in Anomalib datasets using numpy arrays.

    This class combines _VideoInputFields and NumpyItem for video-based anomaly detection.
    It includes video-specific fields and validation methods to ensure proper formatting
    for Anomalib's video-based models.
    """

    @staticmethod
    def _validate_image(image: np.ndarray) -> np.ndarray:
        return image

    @staticmethod
    def _validate_gt_label(gt_label: np.ndarray) -> np.ndarray:
        return gt_label

    @staticmethod
    def _validate_gt_mask(gt_mask: np.ndarray) -> np.ndarray:
        return gt_mask

    @staticmethod
    def _validate_mask_path(mask_path: str) -> str:
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

    @staticmethod
    def _validate_image(image: np.ndarray) -> np.ndarray:
        return image

    @staticmethod
    def _validate_gt_label(gt_label: np.ndarray) -> np.ndarray:
        return gt_label

    @staticmethod
    def _validate_gt_mask(gt_mask: np.ndarray) -> np.ndarray:
        return gt_mask

    @staticmethod
    def _validate_mask_path(mask_path: list[str]) -> list[str]:
        return mask_path

    @staticmethod
    def _validate_anomaly_map(anomaly_map: np.ndarray) -> np.ndarray:
        return anomaly_map
