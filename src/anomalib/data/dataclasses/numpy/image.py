"""Numpy-based image dataclasses for Anomalib."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np

from anomalib.data.dataclasses.generic import BatchIterateMixin, _ImageInputFields
from anomalib.data.dataclasses.numpy.base import NumpyBatch, NumpyItem


@dataclass
class NumpyImageItem(_ImageInputFields[str], NumpyItem):
    """Dataclass for a single image item in Anomalib datasets using numpy arrays.

    This class combines _ImageInputFields and NumpyItem for image-based anomaly detection.
    It includes image-specific fields and validation methods to ensure proper formatting
    for Anomalib's image-based models.

    Examples:
        >>> item = NumpyImageItem(
        ...     image=np.random.rand(224, 224, 3),
        ...     gt_label=np.array(1),
        ...     gt_mask=np.random.rand(224, 224) > 0.5,
        ...     anomaly_map=np.random.rand(224, 224),
        ...     pred_score=np.array(0.7),
        ...     pred_label=np.array(1),
        ...     image_path="path/to/image.jpg"
        ... )

        >>> # Access fields
        >>> image = item.image
        >>> label = item.gt_label
        >>> path = item.image_path
    """

    @staticmethod
    def _validate_image(image: np.ndarray) -> np.ndarray:
        assert image.ndim == 3, f"Expected 3D image, got {image.ndim}D image."
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
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

    @staticmethod
    def _validate_anomaly_map(anomaly_map: np.ndarray | None) -> np.ndarray | None:
        if anomaly_map is None:
            return None
        assert isinstance(anomaly_map, np.ndarray), f"Anomaly map must be a numpy array, got {type(anomaly_map)}."
        assert anomaly_map.ndim in {
            2,
            3,
        }, f"Anomaly map must have shape [H, W] or [1, H, W], got shape {anomaly_map.shape}."
        if anomaly_map.ndim == 3:
            assert (
                anomaly_map.shape[0] == 1
            ), f"Anomaly map with 3 dimensions must have 1 channel, got {anomaly_map.shape[0]}."
            anomaly_map = anomaly_map.squeeze(0)
        return anomaly_map.astype(np.float32)

    @staticmethod
    def _validate_pred_score(pred_score: np.ndarray | None) -> np.ndarray | None:
        if pred_score is None:
            return None
        if pred_score.ndim == 1:
            assert len(pred_score) == 1, f"Expected single value for pred_score, got {len(pred_score)}."
            pred_score = pred_score[0]
        return pred_score

    @staticmethod
    def _validate_pred_mask(pred_mask: np.ndarray) -> np.ndarray:
        return pred_mask

    @staticmethod
    def _validate_pred_label(pred_label: np.ndarray) -> np.ndarray:
        return pred_label

    @staticmethod
    def _validate_image_path(image_path: str) -> str:
        return image_path


@dataclass
class NumpyImageBatch(BatchIterateMixin[NumpyImageItem], _ImageInputFields[list[str]], NumpyBatch):
    """Dataclass for a batch of image items in Anomalib datasets using numpy arrays.

    This class combines BatchIterateMixin, _ImageInputFields, and NumpyBatch for batches
    of image data. It supports batch operations and iteration over individual NumpyImageItems.
    It ensures proper formatting for Anomalib's image-based models.

    Examples:
        >>> batch = NumpyImageBatch(
        ...     image=np.random.rand(32, 224, 224, 3),
        ...     gt_label=np.random.randint(0, 2, (32,)),
        ...     gt_mask=np.random.rand(32, 224, 224) > 0.5,
        ...     anomaly_map=np.random.rand(32, 224, 224),
        ...     pred_score=np.random.rand(32),
        ...     pred_label=np.random.randint(0, 2, (32,)),
        ...     image_path=["path/to/image_{}.jpg".format(i) for i in range(32)]
        ... )

        >>> # Access batch fields
        >>> images = batch.image
        >>> labels = batch.gt_label
        >>> paths = batch.image_path

        >>> # Iterate over items in the batch
        >>> for item in batch:
        ...     process_item(item)
    """

    item_class = NumpyImageItem

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

    @staticmethod
    def _validate_pred_score(pred_score: np.ndarray) -> np.ndarray:
        return pred_score

    @staticmethod
    def _validate_pred_mask(pred_mask: np.ndarray) -> np.ndarray:
        return pred_mask

    @staticmethod
    def _validate_pred_label(pred_label: np.ndarray) -> np.ndarray:
        return pred_label

    @staticmethod
    def _validate_image_path(image_path: list[str]) -> list[str]:
        return image_path
