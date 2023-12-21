"""Binary classification curve (torch and torchmetrics interfaces).

This module implements interfaces for the code in `binclf_curve_numpy.py`. Check its docstring for more details.
"""

from __future__ import annotations

import torch
from torch import Tensor

from . import binclf_curve_numpy

# =========================================== ARGS VALIDATION ===========================================


def _validate_is_tensor(tensor: Tensor, argname: str | None = None) -> None:
    """Validate that `tensor` is a tensor and convert it to a numpy ndarray.

    Validations will preferably happen in ndarray so the numpy code can be reused without torch,
    so often times the Tensor arguments will be converted to ndarray and then validated.
    """
    argname = f"'{argname}'" if argname is not None else "argument"

    if not isinstance(tensor, Tensor):
        msg = f"Expected {argname} to be a tensor, but got {type(tensor)}"
        raise TypeError(msg)


# =========================================== FUNCTIONAL ===========================================


def per_img_binclf_curve(
    anomaly_maps: Tensor,
    masks: Tensor,
    algorithm: str = binclf_curve_numpy.ALGORITHM_NUMBA,
    threshs_choice: str = binclf_curve_numpy.THRESHS_CHOICE_MINMAX_LINSPACE,
    threshs_given: Tensor | None = None,
    num_threshs: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute the binary classification matrix of each image in the batch for multiple thresholds (shared).

    ATTENTION: tensors are converted to numpy arrays and then converted back to tensors.

    Args:
        anomaly_maps (Tensor): Anomaly score maps of shape (N, H, W [, D, ...])
        masks (Tensor): Binary ground truth masks of shape (N, H, W [, D, ...])
        algorithm (str, optional): Algorithm to use. Defaults to ALGORITHM_NUMBA.
        threshs_choice (str, optional): Sequence of thresholds to use. Defaults to THRESH_SEQUENCE_MINMAX_LINSPACE.
        #
        # `threshs_choice`-dependent arguments
        #
        # THRESH_SEQUENCE_GIVEN
        threshs_given (Tensor, optional): Sequence of thresholds to use.
        #
        # THRESH_SEQUENCE_MINMAX_LINSPACE
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

    """
    _validate_is_tensor(anomaly_maps, argname="anomaly_maps")
    anomaly_maps_array = anomaly_maps.detach().cpu().numpy()

    _validate_is_tensor(masks, argname="masks")
    masks_array = masks.detach().cpu().numpy()

    if threshs_given is not None:
        _validate_is_tensor(threshs_given, argname="threshs_given")
        threshs_given_array = threshs_given.detach().cpu().numpy()
    else:
        threshs_given_array = None

    threshs_array, binclf_curves_array = binclf_curve_numpy.per_img_binclf_curve(
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
