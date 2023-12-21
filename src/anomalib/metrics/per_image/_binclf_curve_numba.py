"""Binary classification matrix curve (NUMBA implementation of low level functions).

See docstring of `binclf_curve` or `binclf_curve_numpy` for more details.
"""

import numba
import numpy as np
from numpy import ndarray


@numba.jit(nopython=True)
def binclf_one_curve_numba(scores: ndarray, gts: ndarray, threshs: ndarray) -> ndarray:
    """ONE binary classification matrix at each threshold (NUMBA implementation).

    This does the same as `_binclf_one_curve_python` but with numba using just-in-time compilation.

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
    scores_pos = scores[gts]
    # the sorting is very important for the algorithm to work and the speedup
    scores_pos = np.sort(scores_pos)
    # start counting with lowest th, so everything is predicted as positive (this variable is updated in the loop)
    num_pos = current_count_tp = len(scores_pos)

    tps = np.empty((num_th,), dtype=np.int64)

    # NEGATIVES
    # same thing but for the negative samples
    scores_neg = scores[~gts]
    scores_neg = np.sort(scores_neg)
    num_neg = current_count_fp = len(scores_neg)

    fps = np.empty((num_th,), dtype=np.int64)

    # it will progressively drop the scores that are below the current th
    for thidx, th in enumerate(threshs):
        num_drop = 0
        num_scores = len(scores_pos)
        while num_drop < num_scores and scores_pos[num_drop] < th:  # ! scores_pos !
            num_drop += 1
        # ---
        scores_pos = scores_pos[num_drop:]
        current_count_tp -= num_drop
        tps[thidx] = current_count_tp

        # same with the negatives
        num_drop = 0
        num_scores = len(scores_neg)
        while num_drop < num_scores and scores_neg[num_drop] < th:  # ! scores_neg !
            num_drop += 1
        # ---
        scores_neg = scores_neg[num_drop:]
        current_count_fp -= num_drop
        fps[thidx] = current_count_fp

    fns = num_pos * np.ones((num_th,), dtype=np.int64) - tps
    tns = num_neg * np.ones((num_th,), dtype=np.int64) - fps

    # sequence of dimensions is (threshs, true class, predicted class) (see docstring)
    return np.stack(
        (
            np.stack((tns, fps), axis=-1),
            np.stack((fns, tps), axis=-1),
        ),
        axis=-1,
    ).transpose(0, 2, 1)


@numba.jit(nopython=True, parallel=True)
def binclf_multiple_curves_numba(scores_batch: ndarray, gts_batch: ndarray, threshs: ndarray) -> ndarray:
    """MULTIPLE binary classification matrix at each threshold (NUMBA implementation).

    This does the same as `_binclf_multiple_curves_python` but with numba,
    using parallelization and just-in-time compilation.

    ATTENTION: VALIDATION IS NOT DONE HERE. Make sure to validate the arguments before calling this function.

    Args:
        scores_batch (ndarray): Anomaly scores (N, D,).
        gts_batch (ndarray): Binary (bool) ground truth of shape (N, D,).
        threshs (ndarray): Sequence of thresholds in ascending order (K,).

    Returns:
        ndarray: Binary classification matrix curves (N, K, 2, 2)

        See docstring of `binclf_multiple_curves` for details.
    """
    num_imgs = scores_batch.shape[0]
    num_th = len(threshs)
    ret = np.empty((num_imgs, num_th, 2, 2), dtype=np.int64)
    for imgidx in numba.prange(num_imgs):
        scoremap = scores_batch[imgidx]
        mask = gts_batch[imgidx]
        ret[imgidx] = binclf_one_curve_numba(scoremap, mask, threshs)
    return ret
