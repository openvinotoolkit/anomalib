"""Utils for validating arguments and results.

TODO(jpcbertoldo): Move validations to a common place and reuse them across the codebase.
https://github.com/openvinotoolkit/anomalib/issues/2093
"""

# Original Code
# https://github.com/jpcbertoldo/aupimo
#
# Modified
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import torch
from torch import Tensor

from .utils import images_classes_from_masks

logger = logging.getLogger(__name__)


def is_num_thresholds_gte2(num_thresholds: int) -> None:
    """Validate the number of thresholds is a positive integer >= 2."""
    if not isinstance(num_thresholds, int):
        msg = f"Expected the number of thresholds to be an integer, but got {type(num_thresholds)}"
        raise TypeError(msg)

    if num_thresholds < 2:
        msg = f"Expected the number of thresholds to be larger than 1, but got {num_thresholds}"
        raise ValueError(msg)


def is_same_shape(*args) -> None:
    """Works both for tensors and ndarrays."""
    assert len(args) > 0
    shapes = sorted({tuple(arg.shape) for arg in args})
    if len(shapes) > 1:
        msg = f"Expected arguments to have the same shape, but got {shapes}"
        raise ValueError(msg)


def is_rate(rate: float | int, zero_ok: bool, one_ok: bool) -> None:
    """Validates a rate parameter.

    Args:
        rate (float | int): The rate to be validated.
        zero_ok (bool): Flag indicating if rate can be 0.
        one_ok (bool): Flag indicating if rate can be 1.
    """
    if not isinstance(rate, float | int):
        msg = f"Expected rate to be a float or int, but got {type(rate)}."
        raise TypeError(msg)

    if rate < 0.0 or rate > 1.0:
        msg = f"Expected rate to be in [0, 1], but got {rate}."
        raise ValueError(msg)

    if not zero_ok and rate == 0.0:
        msg = "Rate cannot be 0."
        raise ValueError(msg)

    if not one_ok and rate == 1.0:
        msg = "Rate cannot be 1."
        raise ValueError(msg)


def is_rate_range(bounds: tuple[float, float]) -> None:
    """Validates the range of rates within the bounds.

    Args:
        bounds (tuple[float, float]): The lower and upper bounds of the rates.
    """
    if not isinstance(bounds, tuple):
        msg = f"Expected the bounds to be a tuple, but got {type(bounds)}"
        raise TypeError(msg)

    if len(bounds) != 2:
        msg = f"Expected the bounds to be a tuple of length 2, but got {len(bounds)}"
        raise ValueError(msg)

    lower, upper = bounds
    is_rate(lower, zero_ok=False, one_ok=False)
    is_rate(upper, zero_ok=False, one_ok=True)

    if lower >= upper:
        msg = f"Expected the upper bound to be larger than the lower bound, but got {upper=} <= {lower=}"
        raise ValueError(msg)


def is_valid_threshold(thresholds: Tensor) -> None:
    """Validate that the thresholds are valid and monotonically increasing."""
    if not isinstance(thresholds, Tensor):
        msg = f"Expected thresholds to be an Tensor, but got {type(thresholds)}"
        raise TypeError(msg)

    if thresholds.ndim != 1:
        msg = f"Expected thresholds to be 1D, but got {thresholds.ndim}"
        raise ValueError(msg)

    if not thresholds.dtype.is_floating_point:
        msg = f"Expected thresholds to be of float type, but got Tensor with dtype {thresholds.dtype}"
        raise TypeError(msg)

    # make sure they are strictly increasing
    if not torch.all(torch.diff(thresholds) > 0):
        msg = "Expected thresholds to be strictly increasing, but it is not."
        raise ValueError(msg)


def validate_threshold_bounds(threshold_bounds: tuple[float, float]) -> None:
    if not isinstance(threshold_bounds, tuple):
        msg = f"Expected threshold bounds to be a tuple, but got {type(threshold_bounds)}."
        raise TypeError(msg)

    if len(threshold_bounds) != 2:
        msg = f"Expected threshold bounds to be a tuple of length 2, but got {len(threshold_bounds)}."
        raise ValueError(msg)

    lower, upper = threshold_bounds

    if not isinstance(lower, float):
        msg = f"Expected lower threshold bound to be a float, but got {type(lower)}."
        raise TypeError(msg)

    if not isinstance(upper, float):
        msg = f"Expected upper threshold bound to be a float, but got {type(upper)}."
        raise TypeError(msg)

    if upper <= lower:
        msg = f"Expected the upper bound to be greater than the lower bound, but got {upper} <= {lower}."
        raise ValueError(msg)


def is_anomaly_maps(anomaly_maps: Tensor) -> None:
    if anomaly_maps.ndim != 3:
        msg = f"Expected anomaly maps have 3 dimensions (N, H, W), but got {anomaly_maps.ndim} dimensions"
        raise ValueError(msg)

    if not anomaly_maps.dtype.is_floating_point:
        msg = (
            "Expected anomaly maps to be an floating Tensor with anomaly scores,"
            f" but got Tensor with dtype {anomaly_maps.dtype}"
        )
        raise TypeError(msg)


def is_masks(masks: Tensor) -> None:
    if masks.ndim != 3:
        msg = f"Expected masks have 3 dimensions (N, H, W), but got {masks.ndim} dimensions"
        raise ValueError(msg)

    if masks.dtype == torch.bool:
        pass
    elif masks.dtype.is_floating_point:
        msg = (
            "Expected masks to be an integer or boolean Tensor with ground truth labels, "
            f"but got Tensor with dtype {masks.dtype}"
        )
        raise TypeError(msg)
    else:
        # assumes the type to be (signed or unsigned) integer
        # this will change with the dataclass refactor
        masks_unique_vals = torch.unique(masks)
        if torch.any((masks_unique_vals != 0) & (masks_unique_vals != 1)):
            msg = (
                "Expected masks to be a *binary* Tensor with ground truth labels, "
                f"but got Tensor with unique values {sorted(masks_unique_vals)}"
            )
            raise ValueError(msg)


def is_binclf_curves(binclf_curves: Tensor, valid_thresholds: Tensor | None) -> None:
    if binclf_curves.ndim != 4:
        msg = f"Expected binclf curves to be 4D, but got {binclf_curves.ndim}D"
        raise ValueError(msg)

    if binclf_curves.shape[-2:] != (2, 2):
        msg = f"Expected binclf curves to have shape (..., 2, 2), but got {binclf_curves.shape}"
        raise ValueError(msg)

    if binclf_curves.dtype != torch.int64:
        msg = f"Expected binclf curves to have dtype int64, but got {binclf_curves.dtype}."
        raise TypeError(msg)

    if (binclf_curves < 0).any():
        msg = "Expected binclf curves to have non-negative values, but got negative values."
        raise ValueError(msg)

    neg = binclf_curves[:, :, 0, :].sum(axis=-1)  # (num_images, num_thresholds)

    if (neg != neg[:, :1]).any():
        msg = "Expected binclf curves to have the same number of negatives per image for every thresh."
        raise ValueError(msg)

    pos = binclf_curves[:, :, 1, :].sum(axis=-1)  # (num_images, num_thresholds)

    if (pos != pos[:, :1]).any():
        msg = "Expected binclf curves to have the same number of positives per image for every thresh."
        raise ValueError(msg)

    if valid_thresholds is None:
        return

    if binclf_curves.shape[1] != valid_thresholds.shape[0]:
        msg = (
            "Expected the binclf curves to have as many confusion matrices as the thresholds sequence, "
            f"but got {binclf_curves.shape[1]} and {valid_thresholds.shape[0]}"
        )
        raise RuntimeError(msg)


def is_images_classes(images_classes: Tensor) -> None:
    if images_classes.ndim != 1:
        msg = f"Expected image classes to be 1D, but got {images_classes.ndim}D."
        raise ValueError(msg)

    if images_classes.dtype == torch.bool:
        pass
    elif images_classes.dtype.is_floating_point:
        msg = (
            "Expected image classes to be an integer or boolean Tensor with ground truth labels, "
            f"but got Tensor with dtype {images_classes.dtype}"
        )
        raise TypeError(msg)
    else:
        # assumes the type to be (signed or unsigned) integer
        # this will change with the dataclass refactor
        unique_vals = torch.unique(images_classes)
        if torch.any((unique_vals != 0) & (unique_vals != 1)):
            msg = (
                "Expected image classes to be a *binary* Tensor with ground truth labels, "
                f"but got Tensor with unique values {sorted(unique_vals)}"
            )
            raise ValueError(msg)


def is_rates(rates: Tensor, nan_allowed: bool) -> None:
    if rates.ndim != 1:
        msg = f"Expected rates to be 1D, but got {rates.ndim}D."
        raise ValueError(msg)

    if not rates.dtype.is_floating_point:
        msg = f"Expected rates to have dtype of float type, but got {rates.dtype}."
        raise ValueError(msg)

    isnan_mask = torch.isnan(rates)
    if nan_allowed:
        # if they are all nan, then there is nothing to validate
        if isnan_mask.all():
            return
        valid_values = rates[~isnan_mask]
    elif isnan_mask.any():
        msg = "Expected rates to not contain NaN values, but got NaN values."
        raise ValueError(msg)
    else:
        valid_values = rates

    if (valid_values < 0).any():
        msg = "Expected rates to have values in the interval [0, 1], but got values < 0."
        raise ValueError(msg)

    if (valid_values > 1).any():
        msg = "Expected rates to have values in the interval [0, 1], but got values > 1."
        raise ValueError(msg)


def is_rate_curve(rate_curve: Tensor, nan_allowed: bool, decreasing: bool) -> None:
    is_rates(rate_curve, nan_allowed=nan_allowed)

    diffs = torch.diff(rate_curve)
    diffs_valid = diffs[~torch.isnan(diffs)] if nan_allowed else diffs

    if decreasing and (diffs_valid > 0).any():
        msg = "Expected rate curve to be monotonically decreasing, but got non-monotonically decreasing values."
        raise ValueError(msg)

    if not decreasing and (diffs_valid < 0).any():
        msg = "Expected rate curve to be monotonically increasing, but got non-monotonically increasing values."
        raise ValueError(msg)


def is_per_image_rate_curves(rate_curves: Tensor, nan_allowed: bool, decreasing: bool | None) -> None:
    if rate_curves.ndim != 2:
        msg = f"Expected per-image rate curves to be 2D, but got {rate_curves.ndim}D."
        raise ValueError(msg)

    if not rate_curves.dtype.is_floating_point:
        msg = f"Expected per-image rate curves to have dtype of float type, but got {rate_curves.dtype}."
        raise ValueError(msg)

    isnan_mask = torch.isnan(rate_curves)
    if nan_allowed:
        # if they are all nan, then there is nothing to validate
        if isnan_mask.all():
            return
        valid_values = rate_curves[~isnan_mask]
    elif isnan_mask.any():
        msg = "Expected per-image rate curves to not contain NaN values, but got NaN values."
        raise ValueError(msg)
    else:
        valid_values = rate_curves

    if (valid_values < 0).any():
        msg = "Expected per-image rate curves to have values in the interval [0, 1], but got values < 0."
        raise ValueError(msg)

    if (valid_values > 1).any():
        msg = "Expected per-image rate curves to have values in the interval [0, 1], but got values > 1."
        raise ValueError(msg)

    if decreasing is None:
        return

    diffs = torch.diff(rate_curves, axis=1)
    diffs_valid = diffs[~torch.isnan(diffs)] if nan_allowed else diffs

    if decreasing and (diffs_valid > 0).any():
        msg = (
            "Expected per-image rate curves to be monotonically decreasing, "
            "but got non-monotonically decreasing values."
        )
        raise ValueError(msg)

    if not decreasing and (diffs_valid < 0).any():
        msg = (
            "Expected per-image rate curves to be monotonically increasing, "
            "but got non-monotonically increasing values."
        )
        raise ValueError(msg)


def is_scores_batch(scores_batch: torch.Tensor) -> None:
    """scores_batch (torch.Tensor): floating (N, D)."""
    if not isinstance(scores_batch, torch.Tensor):
        msg = f"Expected `scores_batch` to be an torch.Tensor, but got {type(scores_batch)}"
        raise TypeError(msg)

    if not scores_batch.dtype.is_floating_point:
        msg = (
            "Expected `scores_batch` to be an floating torch.Tensor with anomaly scores_batch,"
            f" but got torch.Tensor with dtype {scores_batch.dtype}"
        )
        raise TypeError(msg)

    if scores_batch.ndim != 2:
        msg = f"Expected `scores_batch` to be 2D, but got {scores_batch.ndim}"
        raise ValueError(msg)


def is_gts_batch(gts_batch: torch.Tensor) -> None:
    """gts_batch (torch.Tensor): boolean (N, D)."""
    if not isinstance(gts_batch, torch.Tensor):
        msg = f"Expected `gts_batch` to be an torch.Tensor, but got {type(gts_batch)}"
        raise TypeError(msg)

    if gts_batch.dtype != torch.bool:
        msg = (
            "Expected `gts_batch` to be an boolean torch.Tensor with anomaly scores_batch,"
            f" but got torch.Tensor with dtype {gts_batch.dtype}"
        )
        raise TypeError(msg)

    if gts_batch.ndim != 2:
        msg = f"Expected `gts_batch` to be 2D, but got {gts_batch.ndim}"
        raise ValueError(msg)


def has_at_least_one_anomalous_image(masks: torch.Tensor) -> None:
    is_masks(masks)
    image_classes = images_classes_from_masks(masks)
    if (image_classes == 1).sum() == 0:
        msg = "Expected at least one ANOMALOUS image, but found none."
        raise ValueError(msg)


def has_at_least_one_normal_image(masks: torch.Tensor) -> None:
    is_masks(masks)
    image_classes = images_classes_from_masks(masks)
    if (image_classes == 0).sum() == 0:
        msg = "Expected at least one NORMAL image, but found none."
        raise ValueError(msg)


def joint_validate_thresholds_shared_fpr(thresholds: torch.Tensor, shared_fpr: torch.Tensor) -> None:
    if thresholds.shape[0] != shared_fpr.shape[0]:
        msg = (
            "Expected `thresholds` and `shared_fpr` to have the same number of elements, "
            f"but got {thresholds.shape[0]} != {shared_fpr.shape[0]}"
        )
        raise ValueError(msg)


def is_per_image_tprs(per_image_tprs: torch.Tensor, image_classes: torch.Tensor) -> None:
    is_images_classes(image_classes)
    # general validations
    is_per_image_rate_curves(
        per_image_tprs,
        nan_allowed=True,  # normal images have NaN TPRs
        decreasing=None,  # not checked here
    )

    # specific to anomalous images
    is_per_image_rate_curves(
        per_image_tprs[image_classes == 1],
        nan_allowed=False,
        decreasing=True,
    )

    # specific to normal images
    normal_images_tprs = per_image_tprs[image_classes == 0]
    if not normal_images_tprs.isnan().all():
        msg = "Expected all normal images to have NaN TPRs, but some have non-NaN values."
        raise ValueError(msg)


def is_per_image_scores(per_image_scores: torch.Tensor) -> None:
    if per_image_scores.ndim != 1:
        msg = f"Expected per-image scores to be 1D, but got {per_image_scores.ndim}D."
        raise ValueError(msg)


def is_image_class(image_class: int) -> None:
    if image_class not in {0, 1}:
        msg = f"Expected image class to be either 0 for 'normal' or 1 for 'anomalous', but got {image_class}."
        raise ValueError(msg)
