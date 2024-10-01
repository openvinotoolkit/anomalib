"""Per-Image Overlap curve (PIMO, pronounced pee-mo) and its area under the curve (AUPIMO).

Details: `anomalib.metrics.per_image.pimo`.
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
    """Compute the Per-IMage Overlap (PIMO, pronounced pee-mo) curves.

    PIMO is a curve of True Positive Rate (TPR) values on each image across multiple anomaly score thresholds.
    The anomaly score thresholds are indexed by a (cross-image shared) value of False Positive Rate (FPR) measure on
    the normal images.

    Details: `anomalib.metrics.per_image.pimo`.

    Args' notation:
        N: number of images
        H: image height
        W: image width
        K: number of thresholds

    Args:
        anomaly_maps: floating point anomaly score maps of shape (N, H, W)
        masks: binary (bool or int) ground truth masks of shape (N, H, W)
        num_thresholds: number of thresholds to compute (K)

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            [0] thresholds of shape (K,) in ascending order
            [1] shared FPR values of shape (K,) in descending order (indices correspond to the thresholds)
            [2] per-image TPR curves of shape (N, K), axis 1 in descending order (indices correspond to the thresholds)
            [3] image classes of shape (N,) with values 0 (normal) or 1 (anomalous)
    """
    # validate the strings are valid
    _validate.is_num_thresholds_gte2(num_thresholds)
    _validate.is_anomaly_maps(anomaly_maps)
    _validate.is_masks(masks)
    _validate.is_same_shape(anomaly_maps, masks)
    _validate.has_at_least_one_anomalous_image(masks)
    _validate.has_at_least_one_normal_image(masks)

    image_classes = images_classes_from_masks(masks)

    # the thresholds are computed here so that they can be restrained to the normal images
    # therefore getting a better resolution in terms of FPR quantization
    # otherwise the function `binclf_curve_numpy.per_image_binclf_curve` would have the range of thresholds
    # computed from all the images (normal + anomalous)
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


# =========================================== AUPIMO ===========================================


def aupimo_scores(
    anomaly_maps: torch.Tensor,
    masks: torch.Tensor,
    num_thresholds: int = 300_000,
    fpr_bounds: tuple[float, float] = (1e-5, 1e-4),
    force: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Compute the PIMO curves and their Area Under the Curve (i.e. AUPIMO) scores.

    Scores are computed from the integration of the PIMO curves within the given FPR bounds, then normalized to [0, 1].
    It can be thought of as the average TPR of the PIMO curves within the given FPR bounds.

    Details: `anomalib.metrics.per_image.pimo`.

    Args' notation:
        N: number of images
        H: image height
        W: image width
        K: number of thresholds

    Args:
        anomaly_maps: floating point anomaly score maps of shape (N, H, W)
        masks: binary (bool or int) ground truth masks of shape (N, H, W)
        num_thresholds: number of thresholds to compute (K)
        fpr_bounds: lower and upper bounds of the FPR integration range
        force: whether to force the computation despite bad conditions

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            [0] thresholds of shape (K,) in ascending order
            [1] shared FPR values of shape (K,) in descending order (indices correspond to the thresholds)
            [2] per-image TPR curves of shape (N, K), axis 1 in descending order (indices correspond to the thresholds)
            [3] image classes of shape (N,) with values 0 (normal) or 1 (anomalous)
            [4] AUPIMO scores of shape (N,) in [0, 1]
            [5] number of points used in the AUC integration
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
            "The lower bound of the shared FPR integration range is not exactly achieved. "
            f"Expected {fpr_lower_bound} but got {fpr_lower_bound_defacto}, which is not within {rtol=}.",
        )

    if not torch.isclose(
        fpr_upper_bound_defacto,
        torch.tensor(fpr_upper_bound, dtype=fpr_upper_bound_defacto.dtype, device=fpr_upper_bound_defacto.device),
        rtol=rtol,
    ):
        logger.warning(
            "The upper bound of the shared FPR integration range is not exactly achieved. "
            f"Expected {fpr_upper_bound} but got {fpr_upper_bound_defacto}, which is not within {rtol=}.",
        )

    # reminder: fpr lower/upper bound is threshold upper/lower bound (reversed)
    thresh_lower_bound_idx = fpr_upper_bound_thresh_idx
    thresh_upper_bound_idx = fpr_lower_bound_thresh_idx

    # deal with edge cases
    if thresh_lower_bound_idx >= thresh_upper_bound_idx:
        msg = (
            "The thresholds corresponding to the given `fpr_bounds` are not valid because "
            "they matched the same threshold or the are in the wrong order. "
            f"FPR upper/lower = threshold lower/upper = {thresh_lower_bound_idx} and {thresh_upper_bound_idx}."
        )
        raise RuntimeError(msg)

    # limit the curves to the integration range [lbound, ubound]
    shared_fpr_bounded: torch.Tensor = shared_fpr[thresh_lower_bound_idx : (thresh_upper_bound_idx + 1)]
    per_image_tprs_bounded: torch.Tensor = per_image_tprs[:, thresh_lower_bound_idx : (thresh_upper_bound_idx + 1)]

    # `shared_fpr` and `tprs` are in descending order; `flip()` reverts to ascending order
    shared_fpr_bounded = torch.flip(shared_fpr_bounded, dims=[0])
    per_image_tprs_bounded = torch.flip(per_image_tprs_bounded, dims=[1])

    # the log's base does not matter because it's a constant factor canceled by normalization factor
    shared_fpr_bounded_log = torch.log(shared_fpr_bounded)

    # deal with edge cases
    invalid_shared_fpr = ~torch.isfinite(shared_fpr_bounded_log)

    if invalid_shared_fpr.all():
        msg = (
            "Cannot compute AUPIMO because the shared fpr integration range is invalid). "
            "Try increasing the number of thresholds."
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
            "Cannot compute AUPIMO because the shared fpr integration range doesn't have enough points. "
            f"Found {num_points_integral} points in the integration range. "
            "Try increasing `num_thresholds`."
        )
        if not force:
            raise RuntimeError(msg)
        msg += " Computation was forced!"
        logger.warning(msg)

    if num_points_integral < 300:
        logger.warning(
            "The AUPIMO may be inaccurate because the shared fpr integration range doesn't have enough points. "
            f"Found {num_points_integral} points in the integration range. "
            "Try increasing `num_thresholds`.",
        )

    aucs: torch.Tensor = torch.trapezoid(per_image_tprs_bounded, x=shared_fpr_bounded_log, axis=1)

    # normalize, then clip(0, 1) makes sure that the values are in [0, 1] in case of numerical errors
    normalization_factor = aupimo_normalizing_factor(fpr_bounds)
    aucs = (aucs / normalization_factor).clip(0, 1)

    return thresholds, shared_fpr, per_image_tprs, image_classes, aucs, num_points_integral


# =========================================== AUX ===========================================


def thresh_at_shared_fpr_level(
    thresholds: torch.Tensor,
    shared_fpr: torch.Tensor,
    fpr_level: float,
) -> tuple[int, float, torch.Tensor]:
    """Return the threshold and its index at the given shared FPR level.

    Three cases are possible:
    - fpr_level == 0: the lowest threshold that achieves 0 FPR is returned
    - fpr_level == 1: the highest threshold that achieves 1 FPR is returned
    - 0 < fpr_level < 1: the threshold that achieves the closest (higher or lower) FPR to `fpr_level` is returned

    Args:
        thresholds: thresholds at which the shared FPR was computed.
        shared_fpr: shared FPR values.
        fpr_level: shared FPR value at which to get the threshold.

    Returns:
        tuple[int, float, float]:
            [0] index of the threshold
            [1] threshold
            [2] the actual shared FPR value at the returned threshold
    """
    _validate.is_valid_threshold(thresholds)
    _validate.is_rate_curve(shared_fpr, nan_allowed=False, decreasing=True)
    _validate.joint_validate_thresholds_shared_fpr(thresholds, shared_fpr)
    _validate.is_rate(fpr_level, zero_ok=True, one_ok=True)

    shared_fpr_min, shared_fpr_max = shared_fpr.min(), shared_fpr.max()

    if fpr_level < shared_fpr_min:
        msg = (
            "Invalid `fpr_level` because it's out of the range of `shared_fpr` = "
            f"[{shared_fpr_min}, {shared_fpr_max}], and got {fpr_level}."
        )
        raise ValueError(msg)

    if fpr_level > shared_fpr_max:
        msg = (
            "Invalid `fpr_level` because it's out of the range of `shared_fpr` = "
            f"[{shared_fpr_min}, {shared_fpr_max}], and got {fpr_level}."
        )
        raise ValueError(msg)

    # fpr_level == 0 or 1 are special case
    # because there may be multiple solutions, and the chosen should their MINIMUM/MAXIMUM respectively
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
    """Constant that normalizes the AUPIMO integral to 0-1 range.

    It is the maximum possible value from the integral in AUPIMO's definition.
    It corresponds to assuming a constant function T_i: thresh --> 1.

    Args:
        fpr_bounds: lower and upper bounds of the FPR integration range.

    Returns:
        float: the normalization factor (>0).
    """
    _validate.is_rate_range(fpr_bounds)
    fpr_lower_bound, fpr_upper_bound = fpr_bounds
    # the log's base must be the same as the one used in the integration!
    return float(np.log(fpr_upper_bound / fpr_lower_bound))
