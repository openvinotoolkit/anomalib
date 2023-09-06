import numba
import numpy as np
from numpy import ndarray


@numba.jit(nopython=True)
def _binclf_curve_numba(scoremap: ndarray, mask: ndarray, thresholds: ndarray):
    """Compute the binary classification matrix curve of a single image for a given sequence of thresholds.

    This does the same as `__binclf_curves_ndarray_itertools` but with numba using just-in-time compilation.

    ATTENTION:
        1. `thresholds` must be sorted in ascending order!
        2. Argument validation is not done here!


    Note: predicted as positive condition is `score >= th`.

    Args:
            D: number of pixels in each image
        scoremap (ndarray): Anomaly score maps of shape (D,),
        mask (ndarray): Binary (bool) ground truth mask of shape (D,),
        thresholds (ndarray): Sequence of T thresholds to compute the binary classification matrix for.

    Returns:
        ndarray: Binary classification matrix of shape (T, 2, 2)
        The last two dimensions are the confusion matrix for each threshold, organized as (true class, predicted class):
            - `tps`: `[... , 1, 1]`
            - `fps`: `[... , 0, 1]`
            - `fns`: `[... , 1, 0]`
            - `tns`: `[... , 0, 0]`

    """

    num_th = len(thresholds)

    # POSITIVES
    scores_pos = scoremap[mask]
    # the sorting is very important for the algorithm to work and the speedup
    scores_pos = np.sort(scores_pos)
    # start counting with lowest th, so everything is predicted as positive (this variable is updated in the loop)
    num_pos = current_count_tp = len(scores_pos)

    tps = np.empty((num_th,), dtype=np.int64)

    # NEGATIVES
    # same thing but for the negative samples
    scores_neg = scoremap[~mask]
    scores_neg = np.sort(scores_neg)
    num_neg = current_count_fp = len(scores_neg)

    fps = np.empty((num_th,), dtype=np.int64)

    # it will progressively drop the scores that are below the current th
    for thidx, th in enumerate(thresholds):
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

    # sequence of dimensions is (thresholds, true class, predicted class)
    # so `tps` is `confmat[:, 1, 1]`, `fps` is `confmat[:, 0, 1]`, etc.
    return np.stack(
        (
            np.stack((tns, fps), axis=-1),
            np.stack((fns, tps), axis=-1),
        ),
        axis=-1,
    ).transpose(0, 2, 1)


@numba.jit(nopython=True, parallel=True)
def _binclf_curves_numba_parallel(scoremaps: ndarray, masks: ndarray, thresholds: ndarray):
    """Generalize the function above to a batch of images by parallelizing the loop over images.

    This has the same role as

    ```
    _binclf_curves_ndarray_itertools = np.vectorize(
        __binclf_curves_ndarray_itertools,
        signature="(n),(n),(k)->(k,2,2)",
    )
    ```

    but it leverages numba's parallelization.

    ATTENTION:
        1. `thresholds` must be sorted in ascending order!
        2. Argument validation is not done here!

    Args:
            N: number of images
            D: number of pixels in each image
        scoremaps (ndarray): Anomaly score maps of shape (N, D),
        masks (ndarray): Binary (bool) ground truth masks of shape (N, D),
        thresholds (ndarray): Sequence of T thresholds to compute the binary classification matrix for.

    Returns:
        ndarray: Binary classification matrix of shape (N, T, 2, 2)
        The last two dimensions are the confusion matrix for each threshold, organized as (true class, predicted class):
            - `tps`: `[... , 1, 1]`
            - `fps`: `[... , 0, 1]`
            - `fns`: `[... , 1, 0]`
            - `tns`: `[... , 0, 0]`
    """
    num_imgs = scoremaps.shape[0]
    num_th = len(thresholds)
    ret = np.empty((num_imgs, num_th, 2, 2), dtype=np.int64)
    for imgidx in numba.prange(num_imgs):
        scoremap = scoremaps[imgidx]
        mask = masks[imgidx]
        ret[imgidx] = _binclf_curve_numba(scoremap, mask, thresholds)
    return ret
