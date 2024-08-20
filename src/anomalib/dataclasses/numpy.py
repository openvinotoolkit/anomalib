"""Dataclasses for numpy data."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .generic import _GenericBatch, _GenericItem, _InputFields, _OutputFields, _VideoInputFields


# torch image outputs
@dataclass
class NumpyImageItem(
    _GenericItem,
    _OutputFields[np.ndarray, np.ndarray],
    _InputFields[np.ndarray, np.ndarray, np.ndarray, Path],
):
    """Dataclass for numpy image output item."""

    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        return image

    def _validate_gt_label(self, gt_label: np.ndarray) -> np.ndarray:
        return gt_label

    def _validate_gt_mask(self, gt_mask: np.ndarray) -> np.ndarray:
        return gt_mask

    def _validate_mask_path(self, mask_path: Path) -> Path:
        return mask_path


@dataclass
class NumpyImageBatch(
    _GenericBatch[NumpyImageItem],
    _OutputFields[np.ndarray, np.ndarray],
    _InputFields[np.ndarray, np.ndarray, np.ndarray, list[Path]],
):
    """Dataclass for numpy image output batch."""

    item_class = NumpyImageItem

    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        return image

    def _validate_gt_label(self, gt_label: np.ndarray) -> np.ndarray:
        return gt_label

    def _validate_gt_mask(self, gt_mask: np.ndarray) -> np.ndarray:
        return gt_mask

    def _validate_mask_path(self, mask_path: list[Path]) -> list[Path]:
        return mask_path


# torch video outputs
@dataclass
class NumpyVideoItem(
    _GenericItem,
    _OutputFields[np.ndarray, np.ndarray],
    _VideoInputFields[np.ndarray, np.ndarray, np.ndarray, Path],
):
    """Dataclass for numpy video output item."""

    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        return image

    def _validate_gt_label(self, gt_label: np.ndarray) -> np.ndarray:
        return gt_label

    def _validate_gt_mask(self, gt_mask: np.ndarray) -> np.ndarray:
        return gt_mask

    def _validate_mask_path(self, mask_path: Path) -> Path:
        return mask_path


@dataclass
class NumpyVideoBatch(
    _GenericBatch[NumpyVideoItem],
    _OutputFields[np.ndarray, np.ndarray],
    _VideoInputFields[np.ndarray, np.ndarray, np.ndarray, list[Path]],
):
    """Dataclass for numpy video output batch."""

    item_class = NumpyVideoItem

    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        return image

    def _validate_gt_label(self, gt_label: np.ndarray) -> np.ndarray:
        return gt_label

    def _validate_gt_mask(self, gt_mask: np.ndarray) -> np.ndarray:
        return gt_mask

    def _validate_mask_path(self, mask_path: list[Path]) -> list[Path]:
        return mask_path
