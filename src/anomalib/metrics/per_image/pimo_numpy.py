"""Per-Image Overlap curve (PIMO, pronounced pee-mo) and its area under the curve (AUPIMO).

# PIMO

PIMO is a curve of True Positive Rate (TPR) values on each image across multiple anomaly score thresholds.
The anomaly score thresholds are indexed by a (shared) valued of False Positive Rate (FPR) measure on the normal images.

Each *anomalous* image has its own curve such that the X-axis is shared by all of them.

At a given threshold:
    X-axis: Shared FPR (may vary)
        1. Log of the Average of per-image FPR on normal images.
        SEE NOTE BELOW.
    Y-axis: per-image TP Rate (TPR), or "Overlap" between the ground truth and the predicted masks.

*** Note about other shared FPR alternatives ***
The shared FPR metric can be made harder by using the cross-image max (or high-percentile) FPRs instead of the mean.
Rationale: this will further punish models that have exceptional FPs in normal images.
So far there is only one shared FPR metric implemented but others will be added in the future.

# AUPIMO

`AUPIMO` is the area under each `PIMO` curve with bounded integration range in terms of shared FPR.

author: jpcbertoldo
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import warnings
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy import ndarray

from . import _validate, binclf_curve_numpy
from .binclf_curve_numpy import BinclfAlgorithm, BinclfThreshsChoice

logger = logging.getLogger(__name__)

# =========================================== CONSTANTS ===========================================


@dataclass
class PIMOSharedFPRMetric:
    """Shared FPR metric (x-axis of the PIMO curve)."""

    MEAN_PERIMAGE_FPR: ClassVar[str] = "mean-per-image-fpr"

    METRICS: ClassVar[tuple[str, ...]] = (MEAN_PERIMAGE_FPR,)

    @staticmethod
    def validate(metric: str) -> None:
        """Validate the argument `metric`."""
        if metric not in PIMOSharedFPRMetric.METRICS:
            msg = f"Invalid `metric`. Expected one of {PIMOSharedFPRMetric.METRICS}, but got {metric} instead."
            raise ValueError(msg)


# =========================================== AUX ===========================================


def _images_classes_from_masks(masks: ndarray) -> ndarray:
    """Deduce the image classes from the masks."""
    _validate.masks(masks)
    return (masks == 1).any(axis=(1, 2)).astype(np.int32)


# =========================================== ARGS VALIDATION ===========================================


def _validate_at_least_one_anomalous_image(masks: ndarray) -> None:
    image_classes = _images_classes_from_masks(masks)
    if (image_classes == 1).sum() == 0:
        msg = "Expected at least one ANOMALOUS image, but found none."
        raise ValueError(msg)


def _validate_at_least_one_normal_image(masks: ndarray) -> None:
    image_classes = _images_classes_from_masks(masks)
    if (image_classes == 0).sum() == 0:
        msg = "Expected at least one NORMAL image, but found none."
        raise ValueError(msg)


def _joint_validate_threshs_shared_fpr(threshs: ndarray, shared_fpr: ndarray) -> None:
    if threshs.shape[0] != shared_fpr.shape[0]:
        msg = (
            "Expected `threshs` and `shared_fpr` to have the same number of elements, "
            f"but got {threshs.shape[0]} != {shared_fpr.shape[0]}"
        )
        raise ValueError(msg)


# =========================================== PIMO ===========================================


def pimo_curves(
    anomaly_maps: ndarray,
    masks: ndarray,
    num_threshs: int,
    binclf_algorithm: str = BinclfAlgorithm.NUMBA,
    shared_fpr_metric: str = PIMOSharedFPRMetric.MEAN_PERIMAGE_FPR,
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """Compute the Per-IMage Overlap (PIMO, pronounced pee-mo) curves.

    PIMO is a curve of True Positive Rate (TPR) values on each image across multiple anomaly score thresholds.
    The anomaly score thresholds are indexed by a (cross-image shared) value of False Positive Rate (FPR) measure on
    the normal images.

    See the module's docstring for more details.

    Args' notation:
        N: number of images
        H: image height
        W: image width
        K: number of thresholds

    Args:
        anomaly_maps: floating point anomaly score maps of shape (N, H, W)
        masks: binary (bool or int) ground truth masks of shape (N, H, W)
        num_threshs: number of thresholds to compute (K)
        binclf_algorithm: algorithm to compute the binary classifier curve (see `binclf_curve_numpy.Algorithm`)
        shared_fpr_metric: metric to compute the shared FPR axis

    Returns:
        tuple[ndarray, ndarray, ndarray, ndarray]:
            [0] thresholds of shape (K,) in ascending order
            [1] shared FPR values of shape (K,) in descending order (indices correspond to the thresholds)
            [2] per-image TPR curves of shape (N, K), axis 1 in descending order (indices correspond to the thresholds)
            [3] image classes of shape (N,) with values 0 (normal) or 1 (anomalous)
    """
    BinclfAlgorithm.validate(binclf_algorithm)
    PIMOSharedFPRMetric.validate(shared_fpr_metric)
    _validate.num_threshs(num_threshs)
    _validate.anomaly_maps(anomaly_maps)
    _validate.masks(masks)
    _validate.same_shape(anomaly_maps, masks)
    _validate_at_least_one_anomalous_image(masks)
    _validate_at_least_one_normal_image(masks)

    image_classes = _images_classes_from_masks(masks)

    # the thresholds are computed here so that they can be restrained to the normal images
    # therefore getting a better resolution in terms of FPR quantization
    # otherwise the function `binclf_curve_numpy.per_image_binclf_curve` would have the range of thresholds
    # computed from all the images (normal + anomalous)
    threshs = binclf_curve_numpy._get_threshs_minmax_linspace(  # noqa: SLF001
        anomaly_maps[image_classes == 0],
        num_threshs,
    )

    # N: number of images, K: number of thresholds
    # shapes are (K,) and (N, K, 2, 2)
    threshs, binclf_curves = binclf_curve_numpy.per_image_binclf_curve(
        anomaly_maps=anomaly_maps,
        masks=masks,
        algorithm=binclf_algorithm,
        threshs_choice=BinclfThreshsChoice.GIVEN,
        threshs_given=threshs,
        num_threshs=None,
    )

    shared_fpr: ndarray
    if shared_fpr_metric == PIMOSharedFPRMetric.MEAN_PERIMAGE_FPR:
        # shape -> (N, K)
        per_image_fprs_normals = binclf_curve_numpy.per_image_fpr(binclf_curves[image_classes == 0])
        try:
            _validate.per_image_rate_curves(per_image_fprs_normals, nan_allowed=False, decreasing=True)
        except ValueError as ex:
            msg = f"Cannot compute PIMO because the per-image FPR curves from normal images are invalid. Cause: {ex}"
            raise RuntimeError(msg) from ex

        # shape -> (K,)
        # this is the only shared FPR metric implemented so far, see note about shared FPR in the module's docstring
        shared_fpr = per_image_fprs_normals.mean(axis=0)

    else:
        msg = f"Shared FPR metric `{shared_fpr_metric}` is not implemented."
        raise NotImplementedError(msg)

    # shape -> (N, K)
    per_image_tprs = binclf_curve_numpy.per_image_tpr(binclf_curves)

    return threshs, shared_fpr, per_image_tprs, image_classes


# =========================================== AUPIMO ===========================================


def aupimo_scores(
    anomaly_maps: ndarray,
    masks: ndarray,
    num_threshs: int = 300_000,
    binclf_algorithm: str = BinclfAlgorithm.NUMBA,
    shared_fpr_metric: str = PIMOSharedFPRMetric.MEAN_PERIMAGE_FPR,
    fpr_bounds: tuple[float, float] = (1e-5, 1e-4),
    force: bool = False,
) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray, int]:
    """Compute the PIMO curves and their Area Under the Curve (i.e. AUPIMO) scores.

    Scores are computed from the integration of the PIMO curves within the given FPR bounds, then normalized to [0, 1].
    It can be thought of as the average TPR of the PIMO curves within the given FPR bounds.

    See `pimo_curves()` and the module's docstring for more details.

    Args' notation:
        N: number of images
        H: image height
        W: image width
        K: number of thresholds

    Args:
        anomaly_maps: floating point anomaly score maps of shape (N, H, W)
        masks: binary (bool or int) ground truth masks of shape (N, H, W)
        num_threshs: number of thresholds to compute (K)
        binclf_algorithm: algorithm to compute the binary classifier curve (see `binclf_curve_numpy.Algorithm`)
        shared_fpr_metric: metric to compute the shared FPR axis
        fpr_bounds: lower and upper bounds of the FPR integration range
        force: whether to force the computation despite bad conditions

    Returns:
        tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
            [0] thresholds of shape (K,) in ascending order
            [1] shared FPR values of shape (K,) in descending order (indices correspond to the thresholds)
            [2] per-image TPR curves of shape (N, K), axis 1 in descending order (indices correspond to the thresholds)
            [3] image classes of shape (N,) with values 0 (normal) or 1 (anomalous)
            [4] AUPIMO scores of shape (N,) in [0, 1]
            [5] number of points used in the AUC integration
    """
    _validate.rate_range(fpr_bounds)

    # other validations are done in the `pimo` function
    threshs, shared_fpr, per_image_tprs, image_classes = pimo_curves(
        anomaly_maps=anomaly_maps,
        masks=masks,
        num_threshs=num_threshs,
        binclf_algorithm=binclf_algorithm,
        shared_fpr_metric=shared_fpr_metric,
    )
    try:
        _validate.threshs(threshs)
        _validate.rate_curve(shared_fpr, nan_allowed=False, decreasing=True)
        _validate.images_classes(image_classes)
        _validate.per_image_rate_curves(per_image_tprs[image_classes == 1], nan_allowed=False, decreasing=True)

    except ValueError as ex:
        msg = f"Cannot compute AUPIMO because the PIMO curves are invalid. Cause: {ex}"
        raise RuntimeError(msg) from ex

    fpr_lower_bound, fpr_upper_bound = fpr_bounds

    # get the threshold indices where the fpr bounds are achieved
    fpr_lower_bound_thresh_idx, _, fpr_lower_bound_defacto = thresh_at_shared_fpr_level(
        threshs,
        shared_fpr,
        fpr_lower_bound,
    )
    fpr_upper_bound_thresh_idx, _, fpr_upper_bound_defacto = thresh_at_shared_fpr_level(
        threshs,
        shared_fpr,
        fpr_upper_bound,
    )

    if not np.isclose(fpr_lower_bound_defacto, fpr_lower_bound, rtol=(rtol := 1e-2)):
        msg = (
            "The lower bound of the shared FPR integration range is not exactly achieved. "
            f"Expected {fpr_lower_bound} but got {fpr_lower_bound_defacto}, which is not within {rtol=}."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=1)

    if not np.isclose(fpr_upper_bound_defacto, fpr_upper_bound, rtol=rtol):
        msg = (
            "The upper bound of the shared FPR integration range is not exactly achieved. "
            f"Expected {fpr_upper_bound} but got {fpr_upper_bound_defacto}, which is not within {rtol=}."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=1)

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
    shared_fpr_bounded: ndarray = shared_fpr[thresh_lower_bound_idx : (thresh_upper_bound_idx + 1)]
    per_image_tprs_bounded: ndarray = per_image_tprs[:, thresh_lower_bound_idx : (thresh_upper_bound_idx + 1)]

    # `shared_fpr` and `tprs` are in descending order; `flip()` reverts to ascending order
    shared_fpr_bounded = np.flip(shared_fpr_bounded)
    per_image_tprs_bounded = np.flip(per_image_tprs_bounded, axis=1)

    # the log's base does not matter because it's a constant factor canceled by normalization factor
    shared_fpr_bounded_log = np.log(shared_fpr_bounded)

    # deal with edge cases
    invalid_shared_fpr = ~np.isfinite(shared_fpr_bounded_log)

    if invalid_shared_fpr.all():
        msg = (
            "Cannot compute AUPIMO because the shared fpr integration range is invalid). "
            "Try increasing the number of thresholds."
        )
        raise RuntimeError(msg)

    if invalid_shared_fpr.any():
        msg = (
            "Some values in the shared fpr integration range are nan. "
            "The AUPIMO will be computed without these values."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=1)
        logger.warning(msg)

        # get rid of nan values by removing them from the integration range
        shared_fpr_bounded_log = shared_fpr_bounded_log[~invalid_shared_fpr]
        per_image_tprs_bounded = per_image_tprs_bounded[:, ~invalid_shared_fpr]

    num_points_integral = int(shared_fpr_bounded_log.shape[0])

    if num_points_integral <= 30:
        msg = (
            "Cannot compute AUPIMO because the shared fpr integration range doesnt have enough points. "
            f"Found {num_points_integral} points in the integration range. "
            "Try increasing `num_threshs`."
        )
        if not force:
            raise RuntimeError(msg)
        msg += " Computation was forced!"
        warnings.warn(msg, RuntimeWarning, stacklevel=1)
        logger.warning(msg)

    if num_points_integral < 300:
        msg = (
            "The AUPIMO may be inaccurate because the shared fpr integration range doesnt have enough points. "
            f"Found {num_points_integral} points in the integration range. "
            "Try increasing `num_threshs`."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=1)
        logger.warning(msg)

    aucs: ndarray = np.trapz(per_image_tprs_bounded, x=shared_fpr_bounded_log, axis=1)

    # normalize, then clip(0, 1) makes sure that the values are in [0, 1] in case of numerical errors
    normalization_factor = aupimo_normalizing_factor(fpr_bounds)
    aucs = (aucs / normalization_factor).clip(0, 1)

    return threshs, shared_fpr, per_image_tprs, image_classes, aucs, num_points_integral


# =========================================== AUX ===========================================


def thresh_at_shared_fpr_level(threshs: ndarray, shared_fpr: ndarray, fpr_level: float) -> tuple[int, float, float]:
    """Return the threshold and its index at the given shared FPR level.

    Three cases are possible:
    - fpr_level == 0: the lowest threshold that achieves 0 FPR is returned
    - fpr_level == 1: the highest threshold that achieves 1 FPR is returned
    - 0 < fpr_level < 1: the threshold that achieves the closest (higher or lower) FPR to `fpr_level` is returned

    Args:
        threshs: thresholds at which the shared FPR was computed.
        shared_fpr: shared FPR values.
        fpr_level: shared FPR value at which to get the threshold.

    Returns:
        tuple[int, float, float]:
            [0] index of the threshold
            [1] threshold
            [2] the actual shared FPR value at the returned threshold
    """
    _validate.threshs(threshs)
    _validate.rate_curve(shared_fpr, nan_allowed=False, decreasing=True)
    _joint_validate_threshs_shared_fpr(threshs, shared_fpr)
    _validate.rate(fpr_level, zero_ok=True, one_ok=True)

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
        index = np.min(np.where(shared_fpr == fpr_level))

    elif fpr_level == 1.0:
        index = np.max(np.where(shared_fpr == fpr_level))

    else:
        index = np.argmin(np.abs(shared_fpr - fpr_level))

    index = int(index)
    fpr_level_defacto = shared_fpr[index]
    thresh = threshs[index]
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
    _validate.rate_range(fpr_bounds)
    fpr_lower_bound, fpr_upper_bound = fpr_bounds
    # the log's base must be the same as the one used in the integration!
    return float(np.log(fpr_upper_bound / fpr_lower_bound))


def aupimo_random_model_score(fpr_bounds: tuple[float, float]) -> float:
    """AUPIMO of a theoretical random model.

    "Random model" means that there is no discrimination between normal and anomalous pixels/patches/images.
    It corresponds to assuming the functions T = F.

    For the FPR bounds (1e-5, 1e-4), the random model AUPIMO is ~4e-5.

    Args:
        fpr_bounds: lower and upper bounds of the FPR integration range.

    Returns:
        float: the AUPIMO score.
    """
    _validate.rate_range(fpr_bounds)
    fpr_lower_bound, fpr_upper_bound = fpr_bounds
    integral_value = fpr_upper_bound - fpr_lower_bound
    return float(integral_value / aupimo_normalizing_factor(fpr_bounds))
