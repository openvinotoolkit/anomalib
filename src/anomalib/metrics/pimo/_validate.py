"""Utilities for validating arguments and results.

This module provides validation functions for various inputs and outputs used in
the PIMO metrics. The functions check for correct data types, shapes, ranges and
other constraints.

The validation functions include:

- Threshold validation (number, bounds, etc)
- Rate validation (ranges, curves, etc)
- Tensor validation (anomaly maps, masks, etc)
- Binary classification curve validation
- Score validation
- Ground truth validation

TODO(jpcbertoldo): Move validations to a common place and reuse them across the
codebase. https://github.com/openvinotoolkit/anomalib/issues/2093

Example:
    >>> from anomalib.metrics.pimo._validate import is_rate
    >>> is_rate(0.5, zero_ok=True, one_ok=True)  # No error
    >>> is_rate(-0.1, zero_ok=True, one_ok=True)  # Raises ValueError
    ValueError: Expected rate to be in [0, 1], but got -0.1.
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
    """Validate that the number of thresholds is a positive integer >= 2.

    Args:
        num_thresholds: Number of thresholds to validate.

    Raises:
        TypeError: If ``num_thresholds`` is not an integer.
        ValueError: If ``num_thresholds`` is less than 2.

    Example:
        >>> is_num_thresholds_gte2(5)  # No error
        >>> is_num_thresholds_gte2(1)  # Raises ValueError
        ValueError: Expected the number of thresholds to be larger than 1, but got 1
    """
    if not isinstance(num_thresholds, int):
        msg = f"Expected the number of thresholds to be an integer, but got {type(num_thresholds)}"
        raise TypeError(msg)

    if num_thresholds < 2:
        msg = f"Expected the number of thresholds to be larger than 1, but got {num_thresholds}"
        raise ValueError(msg)


def is_same_shape(*args) -> None:
    """Validate that all arguments have the same shape.

    Works for both tensors and ndarrays.

    Args:
        *args: Variable number of tensors or ndarrays to compare shapes.

    Raises:
        ValueError: If arguments have different shapes.

    Example:
        >>> import torch
        >>> t1 = torch.zeros(2, 3)
        >>> t2 = torch.ones(2, 3)
        >>> is_same_shape(t1, t2)  # No error
        >>> t3 = torch.zeros(3, 2)
        >>> is_same_shape(t1, t3)  # Raises ValueError
        ValueError: Expected arguments to have the same shape, but got [(2, 3), (3, 2)]
    """
    assert len(args) > 0
    shapes = sorted({tuple(arg.shape) for arg in args})
    if len(shapes) > 1:
        msg = f"Expected arguments to have the same shape, but got {shapes}"
        raise ValueError(msg)


def is_rate(rate: float | int, zero_ok: bool, one_ok: bool) -> None:
    """Validate a rate parameter.

    Args:
        rate: The rate value to validate.
        zero_ok: Whether 0.0 is an acceptable value.
        one_ok: Whether 1.0 is an acceptable value.

    Raises:
        TypeError: If ``rate`` is not a float or int.
        ValueError: If ``rate`` is outside [0,1] or equals 0/1 when not allowed.

    Example:
        >>> is_rate(0.5, zero_ok=True, one_ok=True)  # No error
        >>> is_rate(0.0, zero_ok=False, one_ok=True)  # Raises ValueError
        ValueError: Rate cannot be 0.
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
    """Validate that rate bounds form a valid range.

    Args:
        bounds: Tuple of (lower, upper) rate bounds.

    Raises:
        TypeError: If ``bounds`` is not a tuple of length 2.
        ValueError: If bounds are invalid or lower >= upper.

    Example:
        >>> is_rate_range((0.1, 0.9))  # No error
        >>> is_rate_range((0.9, 0.1))  # Raises ValueError
        ValueError: Expected the upper bound to be larger than the lower bound
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
    """Validate that thresholds are valid and monotonically increasing.

    Args:
        thresholds: Tensor of threshold values.

    Raises:
        TypeError: If ``thresholds`` is not a floating point Tensor.
        ValueError: If ``thresholds`` is not 1D or not strictly increasing.

    Example:
        >>> thresholds = torch.tensor([0.1, 0.2, 0.3])
        >>> is_valid_threshold(thresholds)  # No error
        >>> bad_thresholds = torch.tensor([0.3, 0.2, 0.1])
        >>> is_valid_threshold(bad_thresholds)  # Raises ValueError
        ValueError: Expected thresholds to be strictly increasing
    """
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
    """Validate threshold bounds form a valid range.

    Args:
        threshold_bounds: Tuple of (lower, upper) threshold bounds.

    Raises:
        TypeError: If bounds are not floats or not a tuple of length 2.
        ValueError: If upper <= lower.

    Example:
        >>> validate_threshold_bounds((0.1, 0.9))  # No error
        >>> validate_threshold_bounds((0.9, 0.1))  # Raises ValueError
        ValueError: Expected the upper bound to be greater than the lower bound
    """
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
    """Validate anomaly maps tensor.

    Args:
        anomaly_maps: Tensor of shape (N, H, W) containing anomaly scores.

    Raises:
        ValueError: If tensor does not have 3 dimensions.
        TypeError: If tensor is not floating point.

    Example:
        >>> maps = torch.randn(10, 32, 32)
        >>> is_anomaly_maps(maps)  # No error
        >>> bad_maps = torch.zeros(10, 32, 32, dtype=torch.long)
        >>> is_anomaly_maps(bad_maps)  # Raises TypeError
    """
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
    """Validate ground truth mask tensor.

    Args:
        masks: Binary tensor of shape (N, H, W) containing ground truth labels.

    Raises:
        ValueError: If tensor does not have 3 dimensions or contains non-binary
            values.
        TypeError: If tensor has invalid dtype.

    Example:
        >>> masks = torch.zeros(10, 32, 32, dtype=torch.bool)
        >>> is_masks(masks)  # No error
        >>> bad_masks = torch.ones(10, 32, 32) * 2
        >>> is_masks(bad_masks)  # Raises ValueError
    """
    if masks.ndim != 3:
        msg = f"Expected masks have 3 dimensions (N, H, W), but got {masks.ndim} dimensions"
        raise ValueError(msg)

    if masks.dtype == torch.bool:
        pass
    elif masks.dtype.is_floating_point:
        msg = (
            "Expected masks to be an integer or boolean Tensor with ground truth "
            f"labels, but got Tensor with dtype {masks.dtype}"
        )
        raise TypeError(msg)
    else:
        # assumes the type to be (signed or unsigned) integer
        # this will change with the dataclass refactor
        masks_unique_vals = torch.unique(masks)
        if torch.any((masks_unique_vals != 0) & (masks_unique_vals != 1)):
            msg = (
                "Expected masks to be a *binary* Tensor with ground truth "
                f"labels, but got Tensor with unique values "
                f"{sorted(masks_unique_vals)}"
            )
            raise ValueError(msg)


def is_binclf_curves(
    binclf_curves: Tensor,
    valid_thresholds: Tensor | None,
) -> None:
    """Validate binary classification curves tensor.

    Args:
        binclf_curves: Tensor of shape (N, T, 2, 2) containing confusion matrices
            for N images and T thresholds.
        valid_thresholds: Optional tensor of T threshold values.

    Raises:
        ValueError: If tensor has wrong shape or invalid values.
        TypeError: If tensor has wrong dtype.
        RuntimeError: If number of thresholds doesn't match.

    Example:
        >>> curves = torch.zeros(10, 5, 2, 2, dtype=torch.int64)
        >>> is_binclf_curves(curves, None)  # No error
        >>> bad_curves = torch.zeros(10, 5, 3, 2, dtype=torch.int64)
        >>> is_binclf_curves(bad_curves, None)  # Raises ValueError
    """
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

    neg = binclf_curves[:, :, 0, :].sum(dim=-1)  # (num_images, num_thresholds)

    if (neg != neg[:, :1]).any():
        msg = "Expected binclf curves to have the same number of negatives per image for every thresh."
        raise ValueError(msg)

    pos = binclf_curves[:, :, 1, :].sum(dim=-1)  # (num_images, num_thresholds)

    if (pos != pos[:, :1]).any():
        msg = "Expected binclf curves to have the same number of positives per image for every thresh."
        raise ValueError(msg)

    if valid_thresholds is None:
        return

    if binclf_curves.shape[1] != valid_thresholds.shape[0]:
        msg = (
            "Expected the binclf curves to have as many confusion matrices as "
            f"the thresholds sequence, but got {binclf_curves.shape[1]} and "
            f"{valid_thresholds.shape[0]}"
        )
        raise RuntimeError(msg)


def is_images_classes(images_classes: Tensor) -> None:
    """Validate image-level ground truth labels tensor.

    Args:
        images_classes: Binary tensor of shape (N,) containing image labels.

    Raises:
        ValueError: If tensor is not 1D or contains non-binary values.
        TypeError: If tensor has invalid dtype.

    Example:
        >>> classes = torch.zeros(10, dtype=torch.bool)
        >>> is_images_classes(classes)  # No error
        >>> bad_classes = torch.ones(10) * 2
        >>> is_images_classes(bad_classes)  # Raises ValueError
    """
    if images_classes.ndim != 1:
        msg = f"Expected image classes to be 1D, but got {images_classes.ndim}D."
        raise ValueError(msg)

    if images_classes.dtype == torch.bool:
        pass
    elif images_classes.dtype.is_floating_point:
        msg = (
            "Expected image classes to be an integer or boolean Tensor with "
            f"ground truth labels, but got Tensor with dtype "
            f"{images_classes.dtype}"
        )
        raise TypeError(msg)
    else:
        # assumes the type to be (signed or unsigned) integer
        # this will change with the dataclass refactor
        unique_vals = torch.unique(images_classes)
        if torch.any((unique_vals != 0) & (unique_vals != 1)):
            msg = (
                "Expected image classes to be a *binary* Tensor with ground "
                f"truth labels, but got Tensor with unique values "
                f"{sorted(unique_vals)}"
            )
            raise ValueError(msg)


def is_rates(rates: Tensor, nan_allowed: bool) -> None:
    """Validate rates tensor.

    Args:
        rates: Tensor of shape (N,) containing rate values in [0,1].
        nan_allowed: Whether NaN values are allowed.

    Raises:
        ValueError: If tensor is not 1D, contains values outside [0,1], or has
            NaN when not allowed.
        TypeError: If tensor is not floating point.

    Example:
        >>> rates = torch.tensor([0.1, 0.5, 0.9])
        >>> is_rates(rates, nan_allowed=False)  # No error
        >>> bad_rates = torch.tensor([0.1, float('nan'), 0.9])
        >>> is_rates(bad_rates, nan_allowed=False)  # Raises ValueError
    """
    if rates.ndim != 1:
        msg = f"Expected rates to be 1D, but got {rates.ndim}D."
        raise ValueError(msg)

    if not rates.dtype.is_floating_point:
        msg = f"Expected rates to have dtype of float type, but got {rates.dtype}."
        raise TypeError(msg)

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


def is_rate_curve(
    rate_curve: Tensor,
    nan_allowed: bool,
    decreasing: bool,
) -> None:
    """Validate rate curve tensor.

    Args:
        rate_curve: Tensor of shape (N,) containing rate values.
        nan_allowed: Whether NaN values are allowed.
        decreasing: Whether curve should be monotonically decreasing.

    Raises:
        ValueError: If curve is not monotonic in specified direction.

    Example:
        >>> curve = torch.tensor([0.9, 0.5, 0.1])
        >>> is_rate_curve(curve, nan_allowed=False, decreasing=True)  # No error
        >>> bad_curve = torch.tensor([0.1, 0.5, 0.9])
        >>> is_rate_curve(bad_curve, nan_allowed=False, decreasing=True)
        ValueError: Expected rate curve to be monotonically decreasing
    """
    is_rates(rate_curve, nan_allowed=nan_allowed)

    diffs = torch.diff(rate_curve)
    diffs_valid = diffs[~torch.isnan(diffs)] if nan_allowed else diffs

    if decreasing and (diffs_valid > 0).any():
        msg = "Expected rate curve to be monotonically decreasing, but got non-monotonically decreasing values."
        raise ValueError(msg)

    if not decreasing and (diffs_valid < 0).any():
        msg = "Expected rate curve to be monotonically increasing, but got non-monotonically increasing values."
        raise ValueError(msg)


def is_per_image_rate_curves(
    rate_curves: Tensor,
    nan_allowed: bool,
    decreasing: bool | None,
) -> None:
    """Validate per-image rate curves tensor.

    Args:
        rate_curves: Tensor of shape (N, T) containing rate curves for N images.
        nan_allowed: Whether NaN values are allowed.
        decreasing: Whether curves should be monotonically decreasing.

    Raises:
        ValueError: If curves have invalid values or wrong monotonicity.
        TypeError: If tensor has wrong dtype.

    Example:
        >>> curves = torch.zeros(10, 5)  # 10 images, 5 thresholds
        >>> is_per_image_rate_curves(curves, nan_allowed=False, decreasing=None)
        >>> # No error
    """
    if rate_curves.ndim != 2:
        msg = f"Expected per-image rate curves to be 2D, but got {rate_curves.ndim}D."
        raise ValueError(msg)

    if not rate_curves.dtype.is_floating_point:
        msg = f"Expected per-image rate curves to have dtype of float type, but got {rate_curves.dtype}."
        raise TypeError(msg)

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

    diffs = torch.diff(rate_curves, dim=1)
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
    """Validate batch of anomaly scores.

    Args:
        scores_batch: Floating point tensor of shape (N, D).

    Raises:
        TypeError: If tensor is not floating point.
        ValueError: If tensor is not 2D.

    Example:
        >>> scores = torch.randn(10, 5)  # 10 samples, 5 features
        >>> is_scores_batch(scores)  # No error
        >>> bad_scores = torch.randn(10)  # 1D tensor
        >>> is_scores_batch(bad_scores)  # Raises ValueError
    """
    if not isinstance(scores_batch, torch.Tensor):
        msg = f"Expected `scores_batch` to be an torch.Tensor, but got {type(scores_batch)}"
        raise TypeError(msg)

    if not scores_batch.dtype.is_floating_point:
        msg = (
            "Expected `scores_batch` to be an floating torch.Tensor with "
            f"anomaly scores_batch, but got torch.Tensor with dtype "
            f"{scores_batch.dtype}"
        )
        raise TypeError(msg)

    if scores_batch.ndim != 2:
        msg = f"Expected `scores_batch` to be 2D, but got {scores_batch.ndim}"
        raise ValueError(msg)


def is_gts_batch(gts_batch: torch.Tensor) -> None:
    """Validate batch of ground truth labels.

    Args:
        gts_batch: Boolean tensor of shape (N, D).

    Raises:
        TypeError: If tensor is not boolean.
        ValueError: If tensor is not 2D.

    Example:
        >>> gts = torch.zeros(10, 5, dtype=torch.bool)
        >>> is_gts_batch(gts)  # No error
        >>> bad_gts = torch.zeros(10, dtype=torch.bool)
        >>> is_gts_batch(bad_gts)  # Raises ValueError
    """
    if not isinstance(gts_batch, torch.Tensor):
        msg = f"Expected `gts_batch` to be an torch.Tensor, but got {type(gts_batch)}"
        raise TypeError(msg)

    if gts_batch.dtype != torch.bool:
        msg = (
            "Expected `gts_batch` to be an boolean torch.Tensor with anomaly "
            f"scores_batch, but got torch.Tensor with dtype {gts_batch.dtype}"
        )
        raise TypeError(msg)

    if gts_batch.ndim != 2:
        msg = f"Expected `gts_batch` to be 2D, but got {gts_batch.ndim}"
        raise ValueError(msg)


def has_at_least_one_anomalous_image(masks: torch.Tensor) -> None:
    """Validate presence of at least one anomalous image.

    Args:
        masks: Binary tensor of shape (N, H, W) containing ground truth masks.

    Raises:
        ValueError: If no anomalous images are found.

    Example:
        >>> masks = torch.ones(10, 32, 32, dtype=torch.bool)  # All anomalous
        >>> has_at_least_one_anomalous_image(masks)  # No error
        >>> normal_masks = torch.zeros(10, 32, 32, dtype=torch.bool)
        >>> has_at_least_one_anomalous_image(normal_masks)  # Raises ValueError
    """
    is_masks(masks)
    image_classes = images_classes_from_masks(masks)
    if (image_classes == 1).sum() == 0:
        msg = "Expected at least one ANOMALOUS image, but found none."
        raise ValueError(msg)


def has_at_least_one_normal_image(masks: torch.Tensor) -> None:
    """Validate presence of at least one normal image.

    Args:
        masks: Binary tensor of shape (N, H, W) containing ground truth masks.

    Raises:
        ValueError: If no normal images are found.

    Example:
        >>> masks = torch.zeros(10, 32, 32, dtype=torch.bool)  # All normal
        >>> has_at_least_one_normal_image(masks)  # No error
        >>> anomalous_masks = torch.ones(10, 32, 32, dtype=torch.bool)
        >>> has_at_least_one_normal_image(anomalous_masks)  # Raises ValueError
    """
    is_masks(masks)
    image_classes = images_classes_from_masks(masks)
    if (image_classes == 0).sum() == 0:
        msg = "Expected at least one NORMAL image, but found none."
        raise ValueError(msg)


def joint_validate_thresholds_shared_fpr(
    thresholds: torch.Tensor,
    shared_fpr: torch.Tensor,
) -> None:
    """Validate matching dimensions between thresholds and shared FPR.

    Args:
        thresholds: Tensor of threshold values.
        shared_fpr: Tensor of shared false positive rates.

    Raises:
        ValueError: If tensors have different lengths.

    Example:
        >>> t = torch.linspace(0, 1, 5)
        >>> fpr = torch.zeros(5)
        >>> joint_validate_thresholds_shared_fpr(t, fpr)  # No error
        >>> bad_fpr = torch.zeros(4)
        >>> joint_validate_thresholds_shared_fpr(t, bad_fpr)  # Raises ValueError
    """
    if thresholds.shape[0] != shared_fpr.shape[0]:
        msg = (
            "Expected `thresholds` and `shared_fpr` to have the same number of "
            f"elements, but got {thresholds.shape[0]} != {shared_fpr.shape[0]}"
        )
        raise ValueError(msg)


def is_per_image_tprs(
    per_image_tprs: torch.Tensor,
    image_classes: torch.Tensor,
) -> None:
    """Validate per-image true positive rates.

    Args:
        per_image_tprs: Tensor of TPR values for each image.
        image_classes: Binary tensor indicating normal (0) or anomalous (1)
            images.

    Raises:
        ValueError: If TPRs have invalid values or wrong monotonicity.

    Example:
        >>> tprs = torch.zeros(10, 5)  # 10 images, 5 thresholds
        >>> classes = torch.zeros(10, dtype=torch.bool)
        >>> is_per_image_tprs(tprs, classes)  # No error
    """
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
