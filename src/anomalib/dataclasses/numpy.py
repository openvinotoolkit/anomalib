"""Dataclasses for numpy data."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np

from .generic import BatchIterateMixin, _GenericBatch, _GenericItem, _ImageInputFields, _VideoInputFields


@dataclass
class NumpyItem(_GenericItem[np.ndarray, np.ndarray, np.ndarray, str]):
    """Dataclass for numpy item."""


@dataclass
class NumpyBatch(_GenericBatch[np.ndarray, np.ndarray, np.ndarray, list[str]]):
    """Dataclass for numpy batch."""


# torch image outputs
@dataclass
class NumpyImageItem(
    _ImageInputFields[str],
    NumpyItem,
):
    """Dataclass for numpy image output item."""

    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        assert image.ndim == 3, f"Expected 3D image, got {image.ndim}D image."
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        return image

    def _validate_gt_label(self, gt_label: np.ndarray) -> np.ndarray:
        return gt_label

    def _validate_gt_mask(self, gt_mask: np.ndarray) -> np.ndarray:
        return gt_mask

    def _validate_mask_path(self, mask_path: str) -> str:
        return mask_path

    def _validate_anomaly_map(self, anomaly_map: np.ndarray | None) -> np.ndarray | None:
        if anomaly_map is None:
            return None
        assert isinstance(anomaly_map, np.ndarray), f"Anomaly map must be a numpy array, got {type(anomaly_map)}."
        assert anomaly_map.ndim in [
            2,
            3,
        ], f"Anomaly map must have shape [H, W] or [1, H, W], got shape {anomaly_map.shape}."
        if anomaly_map.ndim == 3:
            assert (
                anomaly_map.shape[0] == 1
            ), f"Anomaly map with 3 dimensions must have 1 channel, got {anomaly_map.shape[0]}."
            anomaly_map = anomaly_map.squeeze(0)
        return anomaly_map.astype(np.float32)

    def _validate_pred_score(self, pred_score: np.ndarray | None) -> np.ndarray | None:
        if pred_score is None:
            return None
        if pred_score.ndim == 1:
            assert len(pred_score) == 1, f"Expected single value for pred_score, got {len(pred_score)}."
            pred_score = pred_score[0]
        return pred_score

    def _validate_pred_mask(self, pred_mask: np.ndarray) -> np.ndarray:
        return pred_mask

    def _validate_pred_label(self, pred_label: np.ndarray) -> np.ndarray:
        return pred_label

    def _validate_image_path(self, image_path: str) -> str:
        return image_path


@dataclass
class NumpyImageBatch(
    BatchIterateMixin[NumpyImageItem],
    _ImageInputFields[list[str]],
    NumpyBatch,
):
    """Dataclass for numpy image output batch."""

    item_class = NumpyImageItem

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


# torch video outputs
@dataclass
class NumpyVideoItem(
    _VideoInputFields[np.ndarray, np.ndarray, np.ndarray, str],
    NumpyItem,
):
    """Dataclass for numpy video output item."""

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
    """Dataclass for numpy video output batch."""

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

    def _validate_pred_score(self, pred_score: np.ndarray) -> np.ndarray:
        return pred_score

    def _validate_pred_mask(self, pred_mask: np.ndarray) -> np.ndarray:
        return pred_mask

    def _validate_pred_label(self, pred_label: np.ndarray) -> np.ndarray:
        return pred_label
