"""Per-Image Overlap curve (PIMO, pronounced pee-mo) and its area under the curve (AUPIMO).

This module provides functions for computing PIMO curves and AUPIMO scores for
anomaly detection evaluation.

The PIMO curve plots True Positive Rate (TPR) values for each image across
multiple anomaly score thresholds. The thresholds are indexed by a shared False
Positive Rate (FPR) measure computed on normal images.

The AUPIMO score is the area under a PIMO curve within specified FPR bounds,
normalized to the range [0,1].

See Also:
    :mod:`anomalib.metrics.per_image.pimo` for detailed documentation.

Example:
    >>> import torch
    >>> anomaly_maps = torch.rand(10, 32, 32)  # 10 images of 32x32
    >>> masks = torch.randint(0, 2, (10, 32, 32))  # Binary masks
    >>> thresholds, shared_fpr, per_image_tprs, classes = pimo_curves(
    ...     anomaly_maps,
    ...     masks,
    ...     num_thresholds=100
    ... )
    >>> aupimo_scores = aupimo_scores(
    ...     anomaly_maps,
    ...     masks,
    ...     num_thresholds=100,
    ...     fpr_bounds=(1e-5, 1e-4)
    ... )
"""

# Original Code
# https://github.com/jpcbertoldo/aupimo
#
# Modified
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np
import torch

from . import _validate
from .binary_classification_curve import (
    ThresholdMethod,
    _get_linspaced_thresholds,
    per_image_fpr,
    per_image_tpr,
    threshold_and_binary_classification_curve,
)
from .utils import images_classes_from_masks

logger = logging.getLogger(__name__)


def pimo_curves(
    anomaly_maps: torch.Tensor,
    masks: torch.Tensor,
    num_thresholds: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the Per-IMage Overlap (PIMO) curves.

    PIMO curves plot True Positive Rate (TPR) values for each image across
    multiple anomaly score thresholds. The thresholds are indexed by a shared
    False Positive Rate (FPR) measure computed on normal images.

    Args:
        anomaly_maps: Anomaly score maps of shape ``(N, H, W)`` where:
            - ``N``: number of images
            - ``H``: image height
            - ``W``: image width
        masks: Binary ground truth masks of shape ``(N, H, W)``
        num_thresholds: Number of thresholds ``K`` to compute

    Returns:
        tuple containing:
            - thresholds: Shape ``(K,)`` in ascending order
            - shared_fpr: Shape ``(K,)`` in descending order
            - per_image_tprs: Shape ``(N, K)`` in descending order
            - image_classes: Shape ``(N,)`` with values 0 (normal) or 1
              (anomalous)

    Raises:
        ValueError: If inputs are invalid or have inconsistent shapes
        RuntimeError: If per-image FPR curves from normal images are invalid

    Example:
        >>> anomaly_maps = torch.rand(10, 32, 32)  # 10 images of 32x32
        >>> masks = torch.randint(0, 2, (10, 32, 32))  # Binary masks
        >>> thresholds, shared_fpr, per_image_tprs, classes = pimo_curves(
        ...     anomaly_maps,
        ...     masks,
        ...     num_thresholds=100
        ... )
    """
    # validate the strings are valid
    _validate.is_num_thresholds_gte2(num_thresholds)
    _validate.is_anomaly_maps(anomaly_maps)
    _validate.is_masks(masks)
    _validate.is_same_shape(anomaly_maps, masks)
    _validate.has_at_least_one_anomalous_image(masks)
    _validate.has_at_least_one_normal_image(masks)

    image_classes = images_classes_from_masks(masks)

    # the thresholds are computed here so that they can be restrained to the
    # normal images therefore getting a better resolution in terms of FPR
    # quantization otherwise the function
    # `binclf_curve_numpy.per_image_binclf_curve` would have the range of
    # thresholds computed from all the images (normal + anomalous)
    thresholds = _get_linspaced_thresholds(
        anomaly_maps[image_classes == 0],
        num_thresholds,
    )

    # N: number of images, K: number of thresholds
    # shapes are (K,) and (N, K, 2, 2)
    thresholds, binclf_curves = threshold_and_binary_classification_curve(
        anomaly_maps=anomaly_maps,
        masks=masks,
        threshold_choice=ThresholdMethod.GIVEN.value,
        thresholds=thresholds,
        num_thresholds=None,
    )

    shared_fpr: torch.Tensor
    # mean-per-image-fpr on normal images
    # shape -> (N, K)
    per_image_fprs_normals = per_image_fpr(binclf_curves[image_classes == 0])
    try:
        _validate.is_per_image_rate_curves(per_image_fprs_normals, nan_allowed=False, decreasing=True)
    except ValueError as ex:
        msg = f"Cannot compute PIMO because the per-image FPR curves from normal images are invalid. Cause: {ex}"
        raise RuntimeError(msg) from ex

    # shape -> (K,)
    # this is the only shared FPR metric implemented so far,
    # see note about shared FPR in Details: `anomalib.metrics.per_image.pimo`.
    shared_fpr = per_image_fprs_normals.mean(axis=0)

    # shape -> (N, K)
    per_image_tprs = per_image_tpr(binclf_curves)

    return thresholds, shared_fpr, per_image_tprs, image_classes


# =========================================== AUPIMO =====================================


def aupimo_scores(
    anomaly_maps: torch.Tensor,
    masks: torch.Tensor,
    num_thresholds: int = 300_000,
    fpr_bounds: tuple[float, float] = (1e-5, 1e-4),
    force: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Compute PIMO curves and their Area Under the Curve (AUPIMO) scores.

    AUPIMO scores are computed by integrating PIMO curves within specified FPR
    bounds and normalizing to [0,1]. The score represents the average TPR within
    the FPR bounds.

    Args:
        anomaly_maps: Anomaly score maps of shape ``(N, H, W)`` where:
            - ``N``: number of images
            - ``H``: image height
            - ``W``: image width
        masks: Binary ground truth masks of shape ``(N, H, W)``
        num_thresholds: Number of thresholds ``K`` to compute
        fpr_bounds: Lower and upper bounds of FPR integration range
        force: Whether to force computation despite bad conditions

    Returns:
        tuple containing:
            - thresholds: Shape ``(K,)`` in ascending order
            - shared_fpr: Shape ``(K,)`` in descending order
            - per_image_tprs: Shape ``(N, K)`` in descending order
            - image_classes: Shape ``(N,)`` with values 0 (normal) or 1
              (anomalous)
            - aupimo_scores: Shape ``(N,)`` in range [0,1]
            - num_points: Number of points used in AUC integration

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If PIMO curves are invalid or integration range has too few
            points

    Example:
        >>> anomaly_maps = torch.rand(10, 32, 32)  # 10 images of 32x32
        >>> masks = torch.randint(0, 2, (10, 32, 32))  # Binary masks
        >>> results = aupimo_scores(
        ...     anomaly_maps,
        ...     masks,
        ...     num_thresholds=100,
        ...     fpr_bounds=(1e-5, 1e-4)
        ... )
        >>> thresholds, shared_fpr, tprs, classes, scores, n_points = results
    """
    _validate.is_rate_range(fpr_bounds)

    # other validations are done in the `pimo` function
    thresholds, shared_fpr, per_image_tprs, image_classes = pimo_curves(
        anomaly_maps=anomaly_maps,
        masks=masks,
        num_thresholds=num_thresholds,
    )
    try:
        _validate.is_valid_threshold(thresholds)
        _validate.is_rate_curve(shared_fpr, nan_allowed=False, decreasing=True)
        _validate.is_images_classes(image_classes)
        _validate.is_per_image_rate_curves(per_image_tprs[image_classes == 1], nan_allowed=False, decreasing=True)

    except ValueError as ex:
        msg = f"Cannot compute AUPIMO because the PIMO curves are invalid. Cause: {ex}"
        raise RuntimeError(msg) from ex

    fpr_lower_bound, fpr_upper_bound = fpr_bounds

    # get the threshold indices where the fpr bounds are achieved
    fpr_lower_bound_thresh_idx, _, fpr_lower_bound_defacto = thresh_at_shared_fpr_level(
        thresholds,
        shared_fpr,
        fpr_lower_bound,
    )
    fpr_upper_bound_thresh_idx, _, fpr_upper_bound_defacto = thresh_at_shared_fpr_level(
        thresholds,
        shared_fpr,
        fpr_upper_bound,
    )

    if not torch.isclose(
        fpr_lower_bound_defacto,
        torch.tensor(fpr_lower_bound, dtype=fpr_lower_bound_defacto.dtype, device=fpr_lower_bound_defacto.device),
        rtol=(rtol := 1e-2),
    ):
        logger.warning(
            "The lower bound of the shared FPR integration range is not exactly "
            f"achieved. Expected {fpr_lower_bound} but got "
            f"{fpr_lower_bound_defacto}, which is not within {rtol=}.",
        )

    if not torch.isclose(
        fpr_upper_bound_defacto,
        torch.tensor(fpr_upper_bound, dtype=fpr_upper_bound_defacto.dtype, device=fpr_upper_bound_defacto.device),
        rtol=rtol,
    ):
        logger.warning(
            "The upper bound of the shared FPR integration range is not exactly "
            f"achieved. Expected {fpr_upper_bound} but got "
            f"{fpr_upper_bound_defacto}, which is not within {rtol=}.",
        )

    # reminder: fpr lower/upper bound is threshold upper/lower bound (reversed)
    thresh_lower_bound_idx = fpr_upper_bound_thresh_idx
    thresh_upper_bound_idx = fpr_lower_bound_thresh_idx

    # deal with edge cases
    if thresh_lower_bound_idx >= thresh_upper_bound_idx:
        msg = (
            "The thresholds corresponding to the given `fpr_bounds` are not "
            "valid because they matched the same threshold or the are in the "
            "wrong order. FPR upper/lower = threshold lower/upper = "
            f"{thresh_lower_bound_idx} and {thresh_upper_bound_idx}."
        )
        raise RuntimeError(msg)

    # limit the curves to the integration range [lbound, ubound]
    shared_fpr_bounded: torch.Tensor = shared_fpr[thresh_lower_bound_idx : (thresh_upper_bound_idx + 1)]
    per_image_tprs_bounded: torch.Tensor = per_image_tprs[:, thresh_lower_bound_idx : (thresh_upper_bound_idx + 1)]

    # `shared_fpr` and `tprs` are in descending order; `flip()` reverts to
    # ascending order
    shared_fpr_bounded = torch.flip(shared_fpr_bounded, dims=[0])
    per_image_tprs_bounded = torch.flip(per_image_tprs_bounded, dims=[1])

    # the log's base does not matter because it's a constant factor canceled by
    # normalization factor
    shared_fpr_bounded_log = torch.log(shared_fpr_bounded)

    # deal with edge cases
    invalid_shared_fpr = ~torch.isfinite(shared_fpr_bounded_log)

    if invalid_shared_fpr.all():
        msg = (
            "Cannot compute AUPIMO because the shared fpr integration range is "
            "invalid). Try increasing the number of thresholds."
        )
        raise RuntimeError(msg)

    if invalid_shared_fpr.any():
        logger.warning(
            "Some values in the shared fpr integration range are nan. "
            "The AUPIMO will be computed without these values.",
        )

        # get rid of nan values by removing them from the integration range
        shared_fpr_bounded_log = shared_fpr_bounded_log[~invalid_shared_fpr]
        per_image_tprs_bounded = per_image_tprs_bounded[:, ~invalid_shared_fpr]

    num_points_integral = int(shared_fpr_bounded_log.shape[0])

    if num_points_integral <= 30:
        msg = (
            "Cannot compute AUPIMO because the shared fpr integration range "
            f"doesn't have enough points. Found {num_points_integral} points in "
            "the integration range. Try increasing `num_thresholds`."
        )
        if not force:
            raise RuntimeError(msg)
        msg += " Computation was forced!"
        logger.warning(msg)

    if num_points_integral < 300:
        logger.warning(
            "The AUPIMO may be inaccurate because the shared fpr integration "
            f"range doesn't have enough points. Found {num_points_integral} "
            "points in the integration range. Try increasing `num_thresholds`.",
        )

    aucs: torch.Tensor = torch.trapezoid(per_image_tprs_bounded, x=shared_fpr_bounded_log, axis=1)

    # normalize, then clip(0, 1) makes sure that the values are in [0, 1] in
    # case of numerical errors
    normalization_factor = aupimo_normalizing_factor(fpr_bounds)
    aucs = (aucs / normalization_factor).clip(0, 1)

    return (thresholds, shared_fpr, per_image_tprs, image_classes, aucs, num_points_integral)


# =========================================== AUX =====================================


def thresh_at_shared_fpr_level(
    thresholds: torch.Tensor,
    shared_fpr: torch.Tensor,
    fpr_level: float,
) -> tuple[int, float, torch.Tensor]:
    """Return the threshold and its index at the given shared FPR level.

    Three cases are possible:
        - ``fpr_level == 0``: lowest threshold achieving 0 FPR is returned
        - ``fpr_level == 1``: highest threshold achieving 1 FPR is returned
        - ``0 < fpr_level < 1``: threshold achieving closest FPR is returned

    Args:
        thresholds: Thresholds at which shared FPR was computed
        shared_fpr: Shared FPR values
        fpr_level: Shared FPR value at which to get threshold

    Returns:
        tuple containing:
            - index: Index of the threshold
            - threshold: Threshold value
            - actual_fpr: Actual shared FPR value at returned threshold

    Raises:
        ValueError: If inputs are invalid or FPR level is out of range

    Example:
        >>> thresholds = torch.linspace(0, 1, 100)
        >>> shared_fpr = torch.linspace(1, 0, 100)  # Decreasing FPR
        >>> idx, thresh, fpr = thresh_at_shared_fpr_level(
        ...     thresholds,
        ...     shared_fpr,
        ...     fpr_level=0.5
        ... )
    """
    _validate.is_valid_threshold(thresholds)
    _validate.is_rate_curve(shared_fpr, nan_allowed=False, decreasing=True)
    _validate.joint_validate_thresholds_shared_fpr(thresholds, shared_fpr)
    _validate.is_rate(fpr_level, zero_ok=True, one_ok=True)

    shared_fpr_min, shared_fpr_max = shared_fpr.min(), shared_fpr.max()

    if fpr_level < shared_fpr_min:
        msg = (
            "Invalid `fpr_level` because it's out of the range of `shared_fpr` "
            f"= [{shared_fpr_min}, {shared_fpr_max}], and got {fpr_level}."
        )
        raise ValueError(msg)

    if fpr_level > shared_fpr_max:
        msg = (
            "Invalid `fpr_level` because it's out of the range of `shared_fpr` "
            f"= [{shared_fpr_min}, {shared_fpr_max}], and got {fpr_level}."
        )
        raise ValueError(msg)

    # fpr_level == 0 or 1 are special case
    # because there may be multiple solutions, and the chosen should their
    # MINIMUM/MAXIMUM respectively
    if fpr_level == 0.0:
        index = torch.min(torch.where(shared_fpr == fpr_level)[0])

    elif fpr_level == 1.0:
        index = torch.max(torch.where(shared_fpr == fpr_level)[0])

    else:
        index = torch.argmin(torch.abs(shared_fpr - fpr_level))

    index = int(index)
    fpr_level_defacto = shared_fpr[index]
    thresh = thresholds[index]
    return index, thresh, fpr_level_defacto


def aupimo_normalizing_factor(fpr_bounds: tuple[float, float]) -> float:
    """Compute constant that normalizes AUPIMO integral to 0-1 range.

    The factor is the maximum possible value from the integral in AUPIMO's
    definition. It corresponds to assuming a constant function T_i: thresh --> 1.

    Args:
        fpr_bounds: Lower and upper bounds of FPR integration range

    Returns:
        float: Normalization factor (>0)

    Example:
        >>> factor = aupimo_normalizing_factor((1e-5, 1e-4))
        >>> print(f"{factor:.3f}")
        2.303
    """
    _validate.is_rate_range(fpr_bounds)
    fpr_lower_bound, fpr_upper_bound = fpr_bounds
    # the log's base must be the same as the one used in the integration!
    return float(np.log(fpr_upper_bound / fpr_lower_bound))
