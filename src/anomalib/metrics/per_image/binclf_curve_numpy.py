"""Binary classification curve (numpy-only implementation).

A binary classification (binclf) matrix (TP, FP, FN, TN) is evaluated at multiple thresholds.

The thresholds are shared by all instances/images, but their binclf are computed independently for each instance/image.
"""

import itertools
import logging

import numpy as np
from numpy import ndarray

try:
    import numba  # noqa: F401
except ImportError:
    HAS_NUMBA = False
else:
    HAS_NUMBA = True
    from . import _binclf_curve_numba


ALGORITHM_PYTHON = "python"
ALGORITHM_NUMBA = "numba"
ALGORIGHTMS = (ALGORITHM_PYTHON, ALGORITHM_NUMBA)

THRESHS_CHOICE_GIVEN = "given"
THRESHS_CHOICE_MINMAX_LINSPACE = "minmax-linspace"
THRESHS_CHOICE_MEAN_FPR_OPTIMIZED = "mean-fpr-optimized"
THRESHS_CHOICES = (
    THRESHS_CHOICE_GIVEN,
    THRESHS_CHOICE_MINMAX_LINSPACE,
    THRESHS_CHOICE_MEAN_FPR_OPTIMIZED,
)


logger = logging.getLogger(__name__)


# =========================================== ARGS VALIDATION ===========================================


def _validate_scores_batch(scores_batch: ndarray) -> None:
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


def _validate_gts_batch(gts_batch: ndarray) -> None:
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


def _validate_threshs(threshs: ndarray) -> None:
    if not isinstance(threshs, ndarray):
        msg = f"Expected `threshs` to be an ndarray, but got {type(threshs)}"
        raise TypeError(msg)

    if threshs.ndim != 1:
        msg = f"Expected `threshs` to be 1D, but got {threshs.ndim}"
        raise ValueError(msg)

    if threshs.dtype.kind != "f":
        msg = (
            "Expected `threshs` to be an floating ndarray with anomaly scores,"
            f" but got ndarray with dtype {threshs.dtype}"
        )
        raise TypeError(msg)

    # make sure they are strictly increasing
    if any(thresh <= prev_th for prev_th, thresh in itertools.pairwise(threshs)):
        msg = "Expected `threshs` to be strictly increasing, but it is not."
        raise ValueError(msg)


def _validate_num_threshs(num_threshs: int) -> None:
    if not isinstance(num_threshs, int):
        msg = f"Expected `num_threshs` to be an integer, but got {type(num_threshs)}"
        raise TypeError(msg)

    if num_threshs < 2:
        msg = f"If argument `num_threshs` is an integer, expected it to be larger than 1, but got {num_threshs}"
        raise ValueError(msg)


def _validate_thresh_bounds(thresh_bounds: tuple[float, float]) -> None:
    if not isinstance(thresh_bounds, tuple):
        msg = f"Expected `thresh_bounds` to be a tuple, but got {type(thresh_bounds)}"
        raise TypeError(msg)

    if len(thresh_bounds) != 2:
        msg = f"Expected `thresh_bounds` to be a tuple of length 2, but got {len(thresh_bounds)}"
        raise ValueError(msg)

    lower, upper = thresh_bounds

    if not isinstance(lower, float) or not isinstance(upper, float):
        msg = f"Expected `thresh_bounds` to be a tuple of floats, but got {type(lower)} and {type(upper)}"
        raise TypeError(msg)

    if lower >= upper:
        msg = f"Expected `thresh_bounds[1]` > `thresh_bounds[0]`, but got {thresh_bounds[1]} <= {thresh_bounds[0]}"
        raise ValueError(msg)


def _validate_anomaly_maps(anomaly_maps: ndarray) -> None:
    if not isinstance(anomaly_maps, ndarray):
        msg = f"Expected `anomaly_maps` to be an ndarray, but got {type(anomaly_maps)}"
        raise TypeError(msg)

    if anomaly_maps.ndim != 3:
        msg = f"Expected `anomaly_maps` have 3 dimensions (N, H, W), but got {anomaly_maps.ndim} dimensions"
        raise ValueError(msg)

    if anomaly_maps.dtype.kind != "f":
        msg = (
            "Expected `anomaly_maps` to be an floating ndarray with anomaly scores,"
            f" but got ndarray with dtype {anomaly_maps.dtype}"
        )
        raise TypeError(msg)


def _validate_masks(masks: ndarray) -> None:
    if not isinstance(masks, ndarray):
        msg = f"Expected `masks` to be an ndarray, but got {type(masks)}"
        raise TypeError(msg)

    if masks.ndim != 3:
        msg = f"Expected `masks` have 3 dimensions (N, H, W), but got {masks.ndim} dimensions"
        raise ValueError(msg)

    if masks.dtype.kind == "b":
        pass

    elif masks.dtype.kind in ("i", "u"):
        masks_unique_vals = np.unique(masks)
        if np.any((masks_unique_vals != 0) & (masks_unique_vals != 1)):
            msg = (
                "Expected `masks` to be a *binary* ndarray with ground truth labels, "
                f"but got ndarray with unique values {masks_unique_vals}"
            )
            raise ValueError(msg)

    else:
        msg = (
            "Expected `masks` to be an integer or boolean ndarray with ground truth labels, "
            f"but got ndarray with dtype {masks.dtype}"
        )
        raise TypeError(msg)


def _validate_same_shape(*args) -> None:
    assert len(args) > 0
    shapes = [tuple(arg.shape) for arg in args]
    if not all(shape == shapes[0] for shape in shapes):
        msg = f"Expecteds to have the same shape, but got {shapes}"
        raise ValueError(msg)


def _validate_binclf_curves(binclf_curves: ndarray, valid_threshs: ndarray | None) -> None:
    if not isinstance(binclf_curves, ndarray):
        msg = f"Expected `binclf_curves` to be an ndarray, but got {type(binclf_curves)}"
        raise TypeError(msg)

    if binclf_curves.ndim != 4:
        msg = f"Expected `binclf_curves` to be 4D, but got {binclf_curves.ndim}D"
        raise ValueError(msg)

    if binclf_curves.shape[-2:] != (2, 2):
        msg = f"Expected `binclf_curves` to have shape (..., 2, 2), but got {binclf_curves.shape}"
        raise ValueError(msg)

    if binclf_curves.dtype != np.int64:
        msg = f"Expected `binclf_curves` to have dtype int64, but got {binclf_curves.dtype}."
        raise TypeError(msg)

    if (binclf_curves < 0).any():
        msg = "Expected `binclf_curves` to have non-negative values, but got negative values."
        raise ValueError(msg)

    neg = binclf_curves[:, :, 0, :].sum(axis=-1)  # (num_images, num_threshs)

    if (neg != neg[:, :1]).any():
        msg = "Expected `binclf_curves` to have the same number of negatives per image for every thresh."
        raise ValueError(msg)

    pos = binclf_curves[:, :, 1, :].sum(axis=-1)  # (num_images, num_threshs)

    if (pos != pos[:, :1]).any():
        msg = "Expected `binclf_curves` to have the same number of positives per image for every thresh."
        raise ValueError(msg)

    if valid_threshs is None:
        return

    if binclf_curves.shape[1] != valid_threshs.shape[0]:
        msg = (
            "Expected `binclf_curves` to have the same number of thresholds as `threshs`, "
            f"but got {binclf_curves.shape[1]} and {valid_threshs.shape[0]}"
        )
        raise RuntimeError(msg)


# =========================================== PYTHON VERSION ===========================================


def _binclf_one_curve_python(scores: ndarray, gts: ndarray, threshs: ndarray) -> ndarray:
    """ONE binary classification matrix at each threshold (PYTHON implementation).

    In the case where the thresholds are given (i.e. not considering all possible thresholds based on the scores),
    this weird-looking function is faster than the two options in `torchmetrics` on the CPU:
        - `_binary_precision_recall_curve_update_vectorized`
        - `_binary_precision_recall_curve_update_loop`

    (both in module `torchmetrics.functional.classification.precision_recall_curve` in `torchmetrics==1.1.0`).

    ATTENTION: VALIDATION IS NOT DONE HERE. Make sure to validate the arguments before calling this function.

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

    def score_less_than_thresh(thresh):  # noqa: ANN001, ANN202
        def func(score) -> bool:  # noqa: ANN001
            return score < thresh

        return func

    # it will progressively drop the scores that are below the current thresh
    for thresh_idx, thresh in enumerate(threshs):
        # UPDATE POSITIVES
        # < becasue it is the same as ~(>=)
        num_drop = sum(1 for _ in itertools.takewhile(score_less_than_thresh(thresh), scores_positives))
        scores_positives = scores_positives[num_drop:]
        current_count_tp -= num_drop
        tps[thresh_idx] = current_count_tp

        # UPDATE NEGATIVES
        # same with the negatives
        num_drop = sum(1 for _ in itertools.takewhile(score_less_than_thresh(thresh), scores_negatives))
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
MULTIPLE binary classification matrix at each threshold (PYTHON implementation).
vectorized version of `_binclf_one_curve_python` (see above)
"""

# =========================================== INTERFACE ===========================================


def binclf_multiple_curves(
    scores_batch: ndarray,
    gts_batch: ndarray,
    threshs: ndarray,
    algorithm: str = ALGORITHM_NUMBA,
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
    """
    _validate_scores_batch(scores_batch)
    _validate_gts_batch(gts_batch)
    _validate_same_shape(scores_batch, gts_batch)
    _validate_threshs(threshs)

    if algorithm == ALGORITHM_PYTHON:
        return _binclf_multiple_curves_python(scores_batch, gts_batch, threshs)

    if algorithm == ALGORITHM_NUMBA:
        if not HAS_NUMBA:
            logger.warning(
                "Algorithm 'numba' was selected, but numba is not installed. Fallback to 'python' algorithm.",
            )
            return _binclf_multiple_curves_python(scores_batch, gts_batch, threshs)
        return _binclf_curve_numba.binclf_multiple_curves_numba(scores_batch, gts_batch, threshs)

    msg = f"Expected `algorithm` to be one of {ALGORIGHTMS}, but got {algorithm}"
    raise NotImplementedError(msg)


# ========================================= PER-IMAGE ===========================================


def per_img_binclf_curve(
    anomaly_maps: ndarray,
    masks: ndarray,
    algorithm: str = ALGORITHM_NUMBA,
    threshs_choice: str = THRESHS_CHOICE_MINMAX_LINSPACE,
    threshs_given: ndarray | None = None,
    num_threshs: int | None = None,
) -> tuple[ndarray, ndarray]:
    """Compute the binary classification matrix of each image in the batch for multiple thresholds (shared).

    Args:
        anomaly_maps (ndarray): Anomaly score maps of shape (N, H, W [, D, ...])
        masks (ndarray): Binary ground truth masks of shape (N, H, W [, D, ...])
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

    """
    # validate inputs
    _validate_anomaly_maps(anomaly_maps)
    _validate_masks(masks)
    _validate_same_shape(anomaly_maps, masks)

    threshs: ndarray

    if threshs_choice == THRESHS_CHOICE_GIVEN:
        assert threshs_given is not None
        _validate_threshs(threshs_given)
        if num_threshs is not None:
            logger.warning(
                f"Argument `num_threshs` was given, but it is ignored because `threshs_choice` is {threshs_choice}.",
            )
        threshs = threshs_given.astype(anomaly_maps.dtype)

    elif threshs_choice == THRESHS_CHOICE_MINMAX_LINSPACE:
        assert num_threshs is not None
        if threshs_given is not None:
            logger.warning(
                f"Argument `threshs_given` was given, but it is ignored because `threshs_choice` is {threshs_choice}.",
            )
        thresh_low, thresh_high = thresh_bounds = (anomaly_maps.min().item(), anomaly_maps.max().item())
        try:
            _validate_thresh_bounds(thresh_bounds)
        except ValueError as ex:
            msg = "Invalid `thresh_bounds` computed from `anomaly_maps`."
            raise ValueError(msg) from ex
        threshs = np.linspace(thresh_low, thresh_high, num_threshs, dtype=anomaly_maps.dtype)

    elif threshs_choice == THRESHS_CHOICE_MEAN_FPR_OPTIMIZED:
        raise NotImplementedError(f"TODO implement {threshs_choice}")  # noqa: EM102

    else:
        msg = f"Expected `threshs_choice` to be one of {THRESHS_CHOICES}, but got {threshs_choice}"
        raise NotImplementedError(msg)

    # keep the batch dimension and flatten the rest
    scores_batch = anomaly_maps.reshape(anomaly_maps.shape[0], -1)
    gts_batch = masks.reshape(masks.shape[0], -1).astype(bool)  # make sure it is boolean

    binclf_curves = binclf_multiple_curves(scores_batch, gts_batch, threshs, algorithm=algorithm)

    num_images = anomaly_maps.shape[0]

    try:
        _validate_binclf_curves(binclf_curves, valid_threshs=threshs)

        # these two validations cannot be done in `_validate_binclf_curves` because it does not have access to the
        # original shapes of `anomaly_maps`
        if binclf_curves.shape[0] != num_images:
            msg = (
                "Expected `binclf_curves` to have the same number of images as `anomaly_maps`, "
                f"but got {binclf_curves.shape[0]} and {anomaly_maps.shape[0]}"
            )
            raise RuntimeError(msg)

    except (TypeError, ValueError) as ex:
        msg = "Invalid `binclf_curves` was computed."
        raise RuntimeError(msg) from ex

    return threshs, binclf_curves
