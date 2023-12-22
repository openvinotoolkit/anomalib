"""Binary classification curve (torch interface).

This module implements torch interfaces to access the numpy code in `binclf_curve_numpy.py`.
Check its docstring for more details.

Tensors are build with `torch.from_numpy` and so the returned tensors will share the same memory as the numpy arrays.

Validations will preferably happen in ndarray so the numpy code can be reused without torch,
so often times the Tensor arguments will be converted to ndarray and then validated.
"""

from __future__ import annotations

import torch
from torch import Tensor

from . import _validate, binclf_curve_numpy
from .binclf_curve_numpy import Algorithm, ThreshsChoice

# =========================================== ARGS VALIDATION ===========================================


def _validate_threshs(threshs: Tensor) -> None:
    _validate.is_tensor(threshs, argname="threshs")
    binclf_curve_numpy._validate_threshs(threshs.detach().cpu().numpy())  # noqa: SLF001


def _validate_binclf_curves(binclf_curves: Tensor, valid_threshs: Tensor | None = None) -> None:
    _validate.is_tensor(binclf_curves, argname="binclf_curves")
    if valid_threshs is not None:
        _validate_threshs(valid_threshs)
    binclf_curve_numpy._validate_binclf_curves(  # noqa: SLF001
        binclf_curves.detach().cpu().numpy(),
        valid_threshs=valid_threshs.detach().cpu().numpy() if valid_threshs is not None else None,
    )


# =========================================== FUNCTIONAL ===========================================


def per_image_binclf_curve(
    anomaly_maps: Tensor,
    masks: Tensor,
    algorithm: str = Algorithm.NUMBA,
    threshs_choice: str = ThreshsChoice.MINMAX_LINSPACE,
    threshs_given: Tensor | None = None,
    num_threshs: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute the binary classification matrix of each image in the batch for multiple thresholds (shared).

    ATTENTION: tensors are converted to numpy arrays and then converted back to tensors (same device as `anomaly_maps`).

    Args:
        anomaly_maps (Tensor): Anomaly score maps of shape (N, H, W [, D, ...])
        masks (Tensor): Binary ground truth masks of shape (N, H, W [, D, ...])
        algorithm (str, optional): Algorithm to use. Defaults to ALGORITHM_NUMBA.
        threshs_choice (str, optional): Sequence of thresholds to use. Defaults to THRESH_SEQUENCE_MINMAX_LINSPACE.
        return_result_object (bool, optional): Whether to return a `PerImageBinClfCurveResult` object. Defaults to True.

        *** `threshs_choice`-dependent arguments ***

        THRESH_SEQUENCE_GIVEN
        threshs_given (Tensor, optional): Sequence of thresholds to use.

        THRESH_SEQUENCE_MINMAX_LINSPACE
        num_threshs (int, optional): Number of thresholds between the min and max of the anomaly maps.

    Returns:
        tuple[Tensor, Tensor]:
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
    _validate.is_tensor(anomaly_maps, argname="anomaly_maps")
    anomaly_maps_array = anomaly_maps.detach().cpu().numpy()

    _validate.is_tensor(masks, argname="masks")
    masks_array = masks.detach().cpu().numpy()

    if threshs_given is not None:
        _validate.is_tensor(threshs_given, argname="threshs_given")
        threshs_given_array = threshs_given.detach().cpu().numpy()
    else:
        threshs_given_array = None

    threshs_array, binclf_curves_array = binclf_curve_numpy.per_image_binclf_curve(
        anomaly_maps=anomaly_maps_array,
        masks=masks_array,
        algorithm=algorithm,
        threshs_choice=threshs_choice,
        threshs_given=threshs_given_array,
        num_threshs=num_threshs,
    )
    threshs = torch.from_numpy(threshs_array).to(anomaly_maps.device)
    binclf_curves = torch.from_numpy(binclf_curves_array).to(anomaly_maps.device).long()

    return threshs, binclf_curves


# =========================================== RATE METRICS ===========================================


def per_image_tpr(binclf_curves: Tensor) -> Tensor:
    """Compute the true positive rates (TPR) for each image in the batch.

    Args:
        binclf_curves (Tensor): Binary classification matrix curves (N, K, 2, 2). See `per_image_binclf_curve`.

    Returns:
        Tensor: True positive rates (TPR) of shape (N, K)

            N: number of images/instances
            K: number of thresholds

            The last dimension is the TPR for each threshold.

            Thresholds are sorted in ascending order, so TPR is in descending order.
    """
    _validate_binclf_curves(binclf_curves)
    binclf_curves_array = binclf_curves.detach().cpu().numpy()
    tprs_array = binclf_curve_numpy.per_image_tpr(binclf_curves_array)
    return torch.from_numpy(tprs_array).to(binclf_curves.device)


def per_image_fpr(binclf_curves: Tensor) -> Tensor:
    """Compute the false positive rates (FPR) for each image in the batch.

    Args:
        binclf_curves (Tensor): Binary classification matrix curves (N, K, 2, 2). See `per_image_binclf_curve`.

    Returns:
        Tensor: False positive rates (FPR) of shape (N, K)

            N: number of images/instances
            K: number of thresholds

            The last dimension is the FPR for each threshold.

            Thresholds are sorted in ascending order, so FPR is in descending order.
    """
    _validate_binclf_curves(binclf_curves)
    binclf_curves_array = binclf_curves.detach().cpu().numpy()
    fprs_array = binclf_curve_numpy.per_image_fpr(binclf_curves_array)
    return torch.from_numpy(fprs_array).to(binclf_curves.device)
