"""Numpy-based image dataclasses for Anomalib."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np

from anomalib.data.dataclasses.generic import BatchIterateMixin, _ImageInputFields
from anomalib.data.dataclasses.numpy.base import NumpyBatch, NumpyItem
from anomalib.data.validators.numpy.image import NumpyImageBatchValidator, NumpyImageValidator


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

    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        return NumpyImageValidator.validate_image(image)

    def _validate_gt_label(self, gt_label: np.ndarray | None) -> np.ndarray | None:
        return NumpyImageValidator.validate_gt_label(gt_label)

    def _validate_gt_mask(self, gt_mask: np.ndarray | None) -> np.ndarray | None:
        return NumpyImageValidator.validate_gt_mask(gt_mask)

    def _validate_mask_path(self, mask_path: str | None) -> str | None:
        return NumpyImageValidator.validate_mask_path(mask_path)

    def _validate_anomaly_map(self, anomaly_map: np.ndarray | None) -> np.ndarray | None:
        return NumpyImageValidator.validate_anomaly_map(anomaly_map)

    def _validate_pred_score(self, pred_score: np.ndarray | None) -> np.ndarray | None:
        return NumpyImageValidator.validate_pred_score(pred_score)

    def _validate_pred_mask(self, pred_mask: np.ndarray | None) -> np.ndarray | None:
        return NumpyImageValidator.validate_pred_mask(pred_mask)

    def _validate_pred_label(self, pred_label: np.ndarray | None) -> np.ndarray | None:
        return NumpyImageValidator.validate_pred_label(pred_label)

    def _validate_image_path(self, image_path: str | None) -> str | None:
        return NumpyImageValidator.validate_image_path(image_path)


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

    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        return NumpyImageBatchValidator.validate_image(image)

    def _validate_gt_label(self, gt_label: np.ndarray | None) -> np.ndarray | None:
        return NumpyImageBatchValidator.validate_gt_label(gt_label, self.batch_size)

    def _validate_gt_mask(self, gt_mask: np.ndarray | None) -> np.ndarray | None:
        return NumpyImageBatchValidator.validate_gt_mask(gt_mask, self.batch_size)

    def _validate_mask_path(self, mask_path: list[str] | None) -> list[str] | None:
        return NumpyImageBatchValidator.validate_mask_path(mask_path, self.batch_size)

    def _validate_anomaly_map(self, anomaly_map: np.ndarray | None) -> np.ndarray | None:
        return NumpyImageBatchValidator.validate_anomaly_map(anomaly_map, self.batch_size)

    def _validate_pred_score(self, pred_score: np.ndarray | None) -> np.ndarray | None:
        return NumpyImageBatchValidator.validate_pred_score(pred_score)

    def _validate_pred_mask(self, pred_mask: np.ndarray | None) -> np.ndarray | None:
        return NumpyImageBatchValidator.validate_pred_mask(pred_mask, self.batch_size)

    def _validate_pred_label(self, pred_label: np.ndarray | None) -> np.ndarray | None:
        return NumpyImageBatchValidator.validate_pred_label(pred_label)

    def _validate_image_path(self, image_path: list[str] | None) -> list[str] | None:
        return NumpyImageBatchValidator.validate_image_path(image_path)
