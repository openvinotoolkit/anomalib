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
    per_image_fpr,
    per_image_tpr,
    threshold_and_binary_classification_curve,
)
from .utils import images_classes_from_masks

logger = logging.getLogger(__name__)


def pimo_curves(
    anomaly_maps: torch.Tensor,
    masks: torch.Tensor,
    fpr_bounds: tuple[float, float] = (1e-5, 1e-4),
    num_thresholds: int = 300,
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
        anomaly_maps: floating point anomaly score maps of shape (N, H, W).
        masks: binary (bool or int) ground truth masks of shape (N, H, W).
        fpr_bounds: lower and upper bounds of the FPR integration range. Default is (1e-5, 1e-4).
        num_thresholds: number of thresholds to compute (K). Default is 300.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            [0] thresholds of shape (K,) in ascending order
            [1] shared FPR values of shape (K,) in descending order (indices correspond to the thresholds)
            [2] per-image TPR curves of shape (N, K), axis 1 in descending order (indices correspond to the thresholds)
            [3] image classes of shape (N,) with values 0 (normal) or 1 (anomalous)
    """
    _validate.is_rate_range(fpr_bounds)
    _validate.is_num_thresholds_gte2(num_thresholds)
    _validate.is_anomaly_maps(anomaly_maps)
    _validate.is_masks(masks)
    _validate.is_same_shape(anomaly_maps, masks)
    _validate.has_at_least_one_anomalous_image(masks)
    _validate.has_at_least_one_normal_image(masks)

    image_classes = images_classes_from_masks(masks)
    anomaly_maps_normal_images = anomaly_maps[image_classes == 0]

    fpr_lower_bound, fpr_upper_bound = fpr_bounds

    # find the thresholds at the given FPR bounds
    threshold_at_fpr_lower_bound = _binary_search_threshold_at_fpr_target(anomaly_maps_normal_images, fpr_lower_bound)
    threshold_at_fpr_upper_bound = _binary_search_threshold_at_fpr_target(anomaly_maps_normal_images, fpr_upper_bound)

    # reminder: fpr lower/upper bound is threshold upper/lower bound (reversed)
    threshold_lower_bound = threshold_at_fpr_upper_bound
    threshold_upper_bound = threshold_at_fpr_lower_bound
    thresholds = torch.linspace(threshold_lower_bound, threshold_upper_bound, num_thresholds, dtype=anomaly_maps.dtype)

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
    fpr_bounds: tuple[float, float] = (1e-5, 1e-4),
    num_thresholds: int = 300,
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
        fpr_bounds: lower and upper bounds of the FPR integration range. Default is (1e-5, 1e-4).
        num_thresholds: number of thresholds to compute (K). Default is 300.
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
    # validations are done in the function `pimo_curves`
    thresholds, shared_fpr, per_image_tprs, image_classes = pimo_curves(
        anomaly_maps=anomaly_maps,
        masks=masks,
        num_thresholds=num_thresholds,
        fpr_bounds=fpr_bounds,
    )

    try:
        _validate.is_valid_threshold(thresholds)
        _validate.is_rate_curve(shared_fpr, nan_allowed=False, decreasing=True)
        _validate.is_images_classes(image_classes)
        _validate.is_per_image_rate_curves(per_image_tprs[image_classes == 1], nan_allowed=False, decreasing=True)

    except ValueError as ex:
        msg = f"Cannot compute AUPIMO because the PIMO curves are invalid. Cause: {ex}"
        raise RuntimeError(msg) from ex

    if num_thresholds < 300:
        logger.warning(
            "The AUPIMO may be inaccurate because the integration range doesn't have enough points. "
            f"Try increasing the values of {num_thresholds=}.",
        )

    fpr_lower_bound, fpr_upper_bound = fpr_bounds

    # get the fpr actual values at the lower/upper bounds of the integration range
    # reminder: fpr lower/upper bound is threshold upper/lower bound (reversed)
    fpr_lower_bound_defacto = shared_fpr[-1]
    fpr_upper_bound_defacto = shared_fpr[0]

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

    # at which threshold the fpr bounds are achieved
    # reminder: fpr lower/upper bound is threshold upper/lower bound (reversed)
    threshold_high_bound = thresholds[-1]  # at fpr lower bound
    threshold_low_bound = thresholds[0]  # at fpr upper bound

    # deal with edge cases
    if threshold_low_bound >= threshold_high_bound:
        msg = (
            "The thresholds corresponding to the given `fpr_bounds` are not valid because "
            "they matched the same threshold or the are in the wrong order. "
            f"FPR upper/lower --> threshold lower|upper = {threshold_low_bound}|{threshold_high_bound}."
        )
        raise RuntimeError(msg)

    # `shared_fpr` and `tprs` are in descending order; `flip()` reverts to ascending order
    # the log's base does not matter because it's a constant factor canceled by normalization factor
    auc_shared_fpr = torch.log(torch.flip(shared_fpr, dims=[0]))
    auc_per_image_tprs = torch.flip(per_image_tprs, dims=[1])

    # deal with edge cases
    invalid_shared_fpr = ~torch.isfinite(auc_shared_fpr)

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
        auc_shared_fpr = auc_shared_fpr[~invalid_shared_fpr]
        auc_per_image_tprs = auc_per_image_tprs[:, ~invalid_shared_fpr]

    # the code above may remove too many points, so we check if there are enough points to integrate
    num_points_integral = int(auc_shared_fpr.shape[0])

    if num_points_integral <= 30:
        msg = (
            "Cannot compute AUPIMO because the shared fpr integration range doesn't have enough points. "
            f"Found {num_points_integral=} points in the integration range. "
            "Try increasing `num_thresholds`."
        )
        if not force:
            raise RuntimeError(msg)
        msg += " Computation was forced!"
        logger.warning(msg)

    if num_points_integral < 300:
        logger.warning(
            "The AUPIMO may be inaccurate because the shared fpr integration range doesn't have enough points. "
            f"Found {num_points_integral=} points in the integration range. "
            "Try increasing `num_thresholds`.",
        )

    aucs: torch.Tensor = torch.trapezoid(auc_per_image_tprs, x=auc_shared_fpr, axis=1)

    # normalize, then clip(0, 1) makes sure that the values are in [0, 1] in case of numerical errors
    normalization_factor = aupimo_normalizing_factor(fpr_bounds)
    aucs = (aucs / normalization_factor).clip(0, 1)

    return thresholds, shared_fpr, per_image_tprs, image_classes, aucs, num_points_integral


# =========================================== AUX ===========================================


def _binary_search_threshold_at_fpr_target(
    anomaly_maps_normals: torch.Tensor,
    fpr_target: float | torch.Tensor,
    maximum_iterations: int = 300,
) -> float:
    """Binary search of threshold that achieves the given shared FPR level.

    Args:
        anomaly_maps_normals: anomaly score maps of normal images.
        fpr_target: shared FPR level at which to get the threshold.
        maximum_iterations: maximum number of iterations for the binary search. Default is 300.

    Returns:
        float: the threshold that achieves the given shared FPR level.
    """
    # binary search bounds
    lower = anomaly_maps_normals.min()
    upper = anomaly_maps_normals.max()
    fpr_target = torch.tensor(fpr_target, dtype=torch.float64)

    # edge case
    if lower == upper:
        return lower.item()

    def get_middle(lower: torch.Tensor, upper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        middle = (lower + upper) / 2
        fpr_at_middle = (anomaly_maps_normals >= middle).double().mean()
        return middle, fpr_at_middle

    for iteration in range(maximum_iterations):  # noqa: B007
        middle, fpr_at_middle = get_middle(lower, upper)

        bounds_are_close = torch.isclose(lower, upper, rtol=1e-6)
        target_is_close = torch.isclose(fpr_at_middle, fpr_target, rtol=1e-2)

        if bounds_are_close and target_is_close:
            break

        # when they are too close, the sign of the difference is not reliable
        # so we make a "half" replacement of the upper/lower bound
        make_big_step = not target_is_close

        if fpr_at_middle < fpr_target:
            upper = middle if make_big_step else (middle + upper) / 2
        else:
            lower = middle if make_big_step else (lower + middle) / 2

    if iteration == maximum_iterations - 1:
        logger.warning(
            f"Binary search reached the maximum number of iterations ({iteration + 1}). "
            "The result may not be accurate. "
            f"Target FPR: {fpr_target:.8g}, achieved FPR: {fpr_at_middle:.8g}. "
            f"Thresholds: {lower=:.8g}, {middle=:.8g}, {upper=:.8g}. "
            f"{bounds_are_close=} {target_is_close=}. "
            f"Try increasing the resolution of the anomaly score maps.",
        )
    else:
        logger.debug(
            f"Binary search stoped with {iteration + 1} iterations. "
            f"Target FPR: {fpr_target:.8g}, achieved FPR: {fpr_at_middle:.8g}. "
            f"Thresholds: {lower=:.8g}, {middle=:.8g}, {upper=:.8g} "
            f"{bounds_are_close=} {target_is_close=}.",
        )

    return middle.item()


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

    if fpr_level < shared_fpr_min and not torch.isclose(shared_fpr_min, torch.tensor(fpr_level).double(), rtol=1e-1):
        msg = (
            "Invalid `fpr_level` because it's out of the range of `shared_fpr` = "
            f"[{shared_fpr_min}, {shared_fpr_max}], and got {fpr_level}."
        )
        raise ValueError(msg)

    if fpr_level > shared_fpr_max and not torch.isclose(shared_fpr_min, torch.tensor(fpr_level).double(), rtol=1e-1):
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
