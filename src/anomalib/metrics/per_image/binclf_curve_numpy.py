"""Binary classification curve (numpy-only implementation).

A binary classification (binclf) matrix (TP, FP, FN, TN) is evaluated at multiple thresholds.

The thresholds are shared by all instances/images, but their binclf are computed independently for each instance/image.

author: jpcbertoldo
"""

# Original Code
# https://github.com/jpcbertoldo/aupimo
#
# Modified
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import itertools
import logging
from enum import Enum
from functools import partial

import numpy as np
from numpy import ndarray

try:
    import numba  # noqa: F401
except ImportError:
    HAS_NUMBA = False
else:
    HAS_NUMBA = True
    from . import _binclf_curve_numba

from . import _validate

logger = logging.getLogger(__name__)

# =========================================== CONSTANTS ===========================================


class BinclfAlgorithm(Enum):
    """Algorithm to use (relates to the low-level implementation)."""

    PYTHON: str = "python"
    NUMBA: str = "numba"


class BinclfThreshsChoice(Enum):
    """Sequence of thresholds to use."""

    GIVEN: str = "given"
    MINMAX_LINSPACE: str = "minmax-linspace"
    MEAN_FPR_OPTIMIZED: str = "mean-fpr-optimized"


# =========================================== ARGS VALIDATION ===========================================


def _validate_is_scores_batch(scores_batch: ndarray) -> None:
    """scores_batch (ndarray): floating (N, D)."""
    if not isinstance(scores_batch, ndarray):
        msg = f"Expected `scores_batch` to be an ndarray, but got {type(scores_batch)}"
        raise TypeError(msg)

    if scores_batch.dtype.kind != "f":
        msg = (
            "Expected `scores_batch` to be an floating ndarray with anomaly scores_batch,"
            f" but got ndarray with dtype {scores_batch.dtype}"
        )
        raise TypeError(msg)

    if scores_batch.ndim != 2:
        msg = f"Expected `scores_batch` to be 2D, but got {scores_batch.ndim}"
        raise ValueError(msg)


def _validate_is_gts_batch(gts_batch: ndarray) -> None:
    """gts_batch (ndarray): boolean (N, D)."""
    if not isinstance(gts_batch, ndarray):
        msg = f"Expected `gts_batch` to be an ndarray, but got {type(gts_batch)}"
        raise TypeError(msg)

    if gts_batch.dtype.kind != "b":
        msg = (
            "Expected `gts_batch` to be an boolean ndarray with anomaly scores_batch,"
            f" but got ndarray with dtype {gts_batch.dtype}"
        )
        raise TypeError(msg)

    if gts_batch.ndim != 2:
        msg = f"Expected `gts_batch` to be 2D, but got {gts_batch.ndim}"
        raise ValueError(msg)


# =========================================== PYTHON VERSION ===========================================


def _binclf_one_curve_python(scores: ndarray, gts: ndarray, threshs: ndarray) -> ndarray:
    """One binary classification matrix at each threshold (PYTHON implementation).

    In the case where the thresholds are given (i.e. not considering all possible thresholds based on the scores),
    this weird-looking function is faster than the two options in `torchmetrics` on the CPU:
        - `_binary_precision_recall_curve_update_vectorized`
        - `_binary_precision_recall_curve_update_loop`

    (both in module `torchmetrics.functional.classification.precision_recall_curve` in `torchmetrics==1.1.0`).

    Note: VALIDATION IS NOT DONE HERE. Make sure to validate the arguments before calling this function.

    Args:
        scores (ndarray): Anomaly scores (D,).
        gts (ndarray): Binary (bool) ground truth of shape (D,).
        threshs (ndarray): Sequence of thresholds in ascending order (K,).

    Returns:
        ndarray: Binary classification matrix curve (K, 2, 2)

        See docstring of `binclf_multiple_curves` for details.
    """
    num_th = len(threshs)

    # POSITIVES
    scores_positives = scores[gts]
    # the sorting is very important for the algorithm to work and the speedup
    scores_positives = np.sort(scores_positives)
    # variable updated in the loop; start counting with lowest thresh ==> everything is predicted as positive
    num_pos = current_count_tp = scores_positives.size
    tps = np.empty((num_th,), dtype=np.int64)

    # NEGATIVES
    # same thing but for the negative samples
    scores_negatives = scores[~gts]
    scores_negatives = np.sort(scores_negatives)
    num_neg = current_count_fp = scores_negatives.size
    fps = np.empty((num_th,), dtype=np.int64)

    def score_less_than_thresh(score: float, thresh: float) -> bool:
        return score < thresh

    # it will progressively drop the scores that are below the current thresh
    for thresh_idx, thresh in enumerate(threshs):
        # UPDATE POSITIVES
        # < becasue it is the same as ~(>=)
        num_drop = sum(1 for _ in itertools.takewhile(partial(score_less_than_thresh, thresh=thresh), scores_positives))
        scores_positives = scores_positives[num_drop:]
        current_count_tp -= num_drop
        tps[thresh_idx] = current_count_tp

        # UPDATE NEGATIVES
        # same with the negatives
        num_drop = sum(1 for _ in itertools.takewhile(partial(score_less_than_thresh, thresh=thresh), scores_negatives))
        scores_negatives = scores_negatives[num_drop:]
        current_count_fp -= num_drop
        fps[thresh_idx] = current_count_fp

    # deduce the rest of the matrix counts
    fns = num_pos * np.ones((num_th,), dtype=np.int64) - tps
    tns = num_neg * np.ones((num_th,), dtype=np.int64) - fps

    # sequence of dimensions is (threshs, true class, predicted class) (see docstring)
    return np.stack(
        [
            np.stack([tns, fps], axis=-1),
            np.stack([fns, tps], axis=-1),
        ],
        axis=-1,
    ).transpose(0, 2, 1)


_binclf_multiple_curves_python = np.vectorize(_binclf_one_curve_python, signature="(n),(n),(k)->(k,2,2)")
_binclf_multiple_curves_python.__doc__ = """
Multiple binary classification matrix at each threshold (PYTHON implementation).
vectorized version of `_binclf_one_curve_python` (see above)
"""

# =========================================== INTERFACE ===========================================


def binclf_multiple_curves(
    scores_batch: ndarray,
    gts_batch: ndarray,
    threshs: ndarray,
    algorithm: BinclfAlgorithm | str = BinclfAlgorithm.NUMBA,
) -> ndarray:
    """Multiple binary classification matrix (per-instance scope) at each threshold (shared).

    This is a wrapper around `_binclf_multiple_curves_python` and `_binclf_multiple_curves_numba`.
    Validation of the arguments is done here (not in the actual implementation functions).

    Note: predicted as positive condition is `score >= thresh`.

    Args:
        scores_batch (ndarray): Anomaly scores (N, D,).
        gts_batch (ndarray): Binary (bool) ground truth of shape (N, D,).
        threshs (ndarray): Sequence of thresholds in ascending order (K,).
        algorithm (str, optional): Algorithm to use. Defaults to ALGORITHM_NUMBA.

    Returns:
        ndarray: Binary classification matrix curves (N, K, 2, 2)

        The last two dimensions are the confusion matrix (ground truth, predictions)
        So for each thresh it gives:
            - `tp`: `[... , 1, 1]`
            - `fp`: `[... , 0, 1]`
            - `fn`: `[... , 1, 0]`
            - `tn`: `[... , 0, 0]`

        `t` is for `true` and `f` is for `false`, `p` is for `positive` and `n` is for `negative`, so:
            - `tp` stands for `true positive`
            - `fp` stands for `false positive`
            - `fn` stands for `false negative`
            - `tn` stands for `true negative`

        The numbers in each confusion matrix are the counts (not the ratios).

        Counts are relative to each instance (i.e. from 0 to D, e.g. the total is the number of pixels in the image).

        Thresholds are shared across all instances, so all confusion matrices, for instance,
        at position [:, 0, :, :] are relative to the 1st threshold in `threshs`.

        Thresholds are sorted in ascending order.
    """
    BinclfAlgorithm(algorithm)
    _validate_is_scores_batch(scores_batch)
    _validate_is_gts_batch(gts_batch)
    _validate.is_same_shape(scores_batch, gts_batch)
    _validate.is_threshs(threshs)

    if BinclfAlgorithm(algorithm) == BinclfAlgorithm.NUMBA:
        if HAS_NUMBA:
            return _binclf_curve_numba.binclf_multiple_curves_numba(scores_batch, gts_batch, threshs)

        logger.warning(
            f"Algorithm '{BinclfAlgorithm.NUMBA.value}' was selected, but numba is not installed. "
            f"Falling back to '{BinclfAlgorithm.PYTHON.value}' implementation.",
        )

    return _binclf_multiple_curves_python(scores_batch, gts_batch, threshs)


# ========================================= PER-IMAGE BINCLF CURVE =========================================


def _get_threshs_minmax_linspace(anomaly_maps: ndarray, num_threshs: int) -> ndarray:
    """Get thresholds linearly spaced between the min and max of the anomaly maps."""
    _validate.is_num_threshs_gte2(num_threshs)
    # this operation can be a bit expensive
    thresh_low, thresh_high = thresh_bounds = (anomaly_maps.min().item(), anomaly_maps.max().item())
    try:
        _validate.is_thresh_bounds(thresh_bounds)
    except ValueError as ex:
        msg = f"Invalid threshold bounds computed from the given anomaly maps. Cause: {ex}"
        raise ValueError(msg) from ex
    return np.linspace(thresh_low, thresh_high, num_threshs, dtype=anomaly_maps.dtype)


def per_image_binclf_curve(
    anomaly_maps: ndarray,
    masks: ndarray,
    algorithm: BinclfAlgorithm | str = BinclfAlgorithm.NUMBA,
    threshs_choice: BinclfThreshsChoice | str = BinclfThreshsChoice.MINMAX_LINSPACE,
    threshs_given: ndarray | None = None,
    num_threshs: int | None = None,
) -> tuple[ndarray, ndarray]:
    """Compute the binary classification matrix of each image in the batch for multiple thresholds (shared).

    Args:
        anomaly_maps (ndarray): Anomaly score maps of shape (N, H, W)
        masks (ndarray): Binary ground truth masks of shape (N, H, W)
        algorithm (str, optional): Algorithm to use. Defaults to ALGORITHM_NUMBA.
        threshs_choice (str, optional): Sequence of thresholds to use. Defaults to THRESH_SEQUENCE_MINMAX_LINSPACE.
        #
        # `threshs_choice`-dependent arguments
        #
        # THRESH_SEQUENCE_GIVEN
        threshs_given (ndarray, optional): Sequence of thresholds to use.
        #
        # THRESH_SEQUENCE_MINMAX_LINSPACE
        num_threshs (int, optional): Number of thresholds between the min and max of the anomaly maps.

    Returns:
        tuple[ndarray, ndarray]:
            [0] Thresholds of shape (K,) and dtype is the same as `anomaly_maps.dtype`.

            [1] Binary classification matrices of shape (N, K, 2, 2)

                N: number of images/instances
                K: number of thresholds

            The last two dimensions are the confusion matrix (ground truth, predictions)
            So for each thresh it gives:
                - `tp`: `[... , 1, 1]`
                - `fp`: `[... , 0, 1]`
                - `fn`: `[... , 1, 0]`
                - `tn`: `[... , 0, 0]`

            `t` is for `true` and `f` is for `false`, `p` is for `positive` and `n` is for `negative`, so:
                - `tp` stands for `true positive`
                - `fp` stands for `false positive`
                - `fn` stands for `false negative`
                - `tn` stands for `true negative`

            The numbers in each confusion matrix are the counts of pixels in the image (not the ratios).

            Thresholds are shared across all images, so all confusion matrices, for instance,
            at position [:, 0, :, :] are relative to the 1st threshold in `threshs`.

            Thresholds are sorted in ascending order.
    """
    BinclfAlgorithm(algorithm)
    _validate.is_anomaly_maps(anomaly_maps)
    _validate.is_masks(masks)
    _validate.is_same_shape(anomaly_maps, masks)

    threshs: ndarray

    if BinclfThreshsChoice(threshs_choice) == BinclfThreshsChoice.GIVEN:
        assert threshs_given is not None
        _validate.is_threshs(threshs_given)
        if num_threshs is not None:
            logger.warning(
                f"Argument `num_threshs` was given, but it is ignored because `threshs_choice` is {threshs_choice}.",
            )
        threshs = threshs_given.astype(anomaly_maps.dtype)

    elif BinclfThreshsChoice(threshs_choice) == BinclfThreshsChoice.MINMAX_LINSPACE:
        assert num_threshs is not None
        if threshs_given is not None:
            logger.warning(
                f"Argument `threshs_given` was given, but it is ignored because `threshs_choice` is {threshs_choice}.",
            )
        # `num_threshs` is validated in the function below
        threshs = _get_threshs_minmax_linspace(anomaly_maps, num_threshs)

    elif BinclfThreshsChoice(threshs_choice) == BinclfThreshsChoice.MEAN_FPR_OPTIMIZED:
        raise NotImplementedError(f"TODO implement {threshs_choice}")  # noqa: EM102

    else:
        msg = f"Expected `threshs_choice` to be from {list(BinclfThreshsChoice.__members__)}, but got {threshs_choice}"
        raise NotImplementedError(msg)

    # keep the batch dimension and flatten the rest
    scores_batch = anomaly_maps.reshape(anomaly_maps.shape[0], -1)
    gts_batch = masks.reshape(masks.shape[0], -1).astype(bool)  # make sure it is boolean

    binclf_curves = binclf_multiple_curves(scores_batch, gts_batch, threshs, algorithm=algorithm)

    num_images = anomaly_maps.shape[0]

    try:
        _validate.is_binclf_curves(binclf_curves, valid_threshs=threshs)

        # these two validations cannot be done in `_validate.binclf_curves` because it does not have access to the
        # original shapes of `anomaly_maps`
        if binclf_curves.shape[0] != num_images:
            msg = (
                "Expected `binclf_curves` to have the same number of images as `anomaly_maps`, "
                f"but got {binclf_curves.shape[0]} and {anomaly_maps.shape[0]}"
            )
            raise RuntimeError(msg)

    except (TypeError, ValueError) as ex:
        msg = f"Invalid `binclf_curves` was computed. Cause: {ex}"
        raise RuntimeError(msg) from ex

    return threshs, binclf_curves


# =========================================== RATE METRICS ===========================================


def per_image_tpr(binclf_curves: ndarray) -> ndarray:
    """True positive rates (TPR) for image for each thresh.

    TPR = TP / P = TP / (TP + FN)

    TP: true positives
    FM: false negatives
    P: positives (TP + FN)

    Args:
        binclf_curves (ndarray): Binary classification matrix curves (N, K, 2, 2). See `per_image_binclf_curve`.

    Returns:
        ndarray: shape (N, K), dtype float64
        N: number of images
        K: number of thresholds

        Thresholds are sorted in ascending order, so TPR is in descending order.
    """
    # shape: (num images, num threshs)
    tps = binclf_curves[..., 1, 1]
    pos = binclf_curves[..., 1, :].sum(axis=2)  # 2 was the 3 originally

    # tprs will be nan if pos == 0 (normal image), which is expected
    return tps.astype(np.float64) / pos.astype(np.float64)


def per_image_fpr(binclf_curves: ndarray) -> ndarray:
    """False positive rates (TPR) for image for each thresh.

    FPR = FP / N = FP / (FP + TN)

    FP: false positives
    TN: true negatives
    N: negatives (FP + TN)

    Args:
        binclf_curves (ndarray): Binary classification matrix curves (N, K, 2, 2). See `per_image_binclf_curve`.

    Returns:
        ndarray: shape (N, K), dtype float64
        N: number of images
        K: number of thresholds

        Thresholds are sorted in ascending order, so FPR is in descending order.
    """
    # shape: (num images, num threshs)
    fps = binclf_curves[..., 0, 1]
    neg = binclf_curves[..., 0, :].sum(axis=2)  # 2 was the 3 originally

    # it can be `nan` if an anomalous image is fully covered by the mask
    return fps.astype(np.float64) / neg.astype(np.float64)
