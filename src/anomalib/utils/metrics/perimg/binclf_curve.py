"""Per-Image Binary Classification Curve.

Binary classification (threshold-dependent) matrix with shared thresholds but per-image counts/rates.

Known issue:

Computing the binary classification matrix curve depends on knowing the min and max anomaly scores across all images,
and the current approach is to just stock all anomaly maps and masks in memory and compute the min and max at the end;
a better approach would be to do the computation in two phases/epochs:
    1. compute the min and max anomaly scores across all images (no need to stock the anomaly maps and masks)
    2. do the actual computation of the binary classification matrix curve, which can actually be done in batches
        once the thresholds are known
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

# =========================================== ARGS VALIDATION ===========================================


def _validate_num_thresholds(num_thresholds: int) -> None:
    NUM_THRESHOLDS_WARNING_LOW = 1_000

    if not isinstance(num_thresholds, int):
        raise ValueError(f"Expected argument `num_thresholds` to be an integer, but got {type(num_thresholds)}")

    if num_thresholds < 2:
        raise ValueError(
            f"If argument `num_thresholds` is an integer, expected it to be larger than 1, but got {num_thresholds}"
        )

    elif num_thresholds < NUM_THRESHOLDS_WARNING_LOW:
        warnings.warn(
            f"Argument `num_thresholds` ({num_thresholds}) is lower than {NUM_THRESHOLDS_WARNING_LOW}. "
            "This may lead to inaccurate results."
        )


def _validate_threshold_bounds(threshold_bounds: Tensor) -> None:
    if not isinstance(threshold_bounds, Tensor):
        raise ValueError(f"Expected argument `threshold_bounds` to be a tensor, but got {type(threshold_bounds)}")

    if not threshold_bounds.is_floating_point():
        raise ValueError(
            "Expected argument `threshold_bounds` to be an floating tensor with anomaly scores,"
            f" but got tensor with dtype {threshold_bounds.dtype}"
        )

    if threshold_bounds.ndim != 1 or threshold_bounds.shape[0] != 2:
        raise ValueError(
            "Expected argument `threshold_bounds` to be a 1D tensor with 2 elements, "
            f"but got {threshold_bounds.shape}"
        )

    if threshold_bounds[0] >= threshold_bounds[1]:
        raise ValueError(
            "Expected argument `threshold_bounds[1]` > `threshold_bounds[0]`, "
            f"but got {threshold_bounds[1]} <= {threshold_bounds[0]}"
        )


def _validate_anomaly_maps(anomaly_maps: Tensor) -> None:
    if anomaly_maps.ndim < 2:
        raise ValueError(f"Expected argument `anomaly_maps` to be at least 2D, but got {anomaly_maps.ndim}")

    if not anomaly_maps.is_floating_point():
        raise ValueError(
            "Expected argument `anomaly_maps` to be an floating tensor with anomaly scores,"
            f" but got tensor with dtype {anomaly_maps.dtype}"
        )


def _validate_masks(masks: Tensor) -> None:
    if masks.ndim < 2:
        raise ValueError(f"Expected argument `masks` to be at least 2D, but got {masks.ndim}")

    if masks.is_floating_point():
        raise ValueError(
            "Expected argument `masks` to be an int or long tensor with ground truth labels"
            f" but got tensor with dtype {masks.dtype}"
        )

    # check that masks are binary
    masks_unique_vals = torch.unique(masks)
    if torch.any((masks_unique_vals != 0) & (masks_unique_vals != 1)):
        raise ValueError(
            "Expected argument `masks` to be a *binary* tensor with ground truth labels, "
            f"but got tensor with unique values {masks_unique_vals}"
        )


def _validate_anomaly_maps_and_masks(anomaly_maps: Tensor, masks: Tensor) -> None:
    _validate_masks(masks)
    _validate_anomaly_maps(anomaly_maps)

    # anomaly_maps and masks must have the same shape
    if anomaly_maps.shape != masks.shape:
        raise ValueError(
            "Expected arguments `anomaly_maps` and `masks` to have the same shape, "
            f"but got {anomaly_maps.shape} and {masks.shape}."
        )


def _validate_tensor_in_cpu(tensor: Tensor) -> None:
    if not tensor.device.type == "cpu":
        raise ValueError(f"Expected argument `tensor` to be on cpu, but got {tensor.device.type}")


def _validated_binclf_curves(binclf_curves: Tensor):
    if not isinstance(binclf_curves, Tensor):
        raise ValueError(f"Expected argument `binclf_curves` to be a Tensor, but got {type(binclf_curves)}.")

    if binclf_curves.ndim != 4:
        raise ValueError(f"Expected argument `binclf_curves` to be a 4D tensor, but got {binclf_curves.ndim}D tensor.")

    if binclf_curves.shape[-2:] != (2, 2):
        raise ValueError(f"Expected argument `binclf_curves` to have shape (..., 2, 2), but got {binclf_curves.shape}.")

    if binclf_curves.dtype != torch.int64:
        raise ValueError(f"Expected argument `binclf_curves` to have dtype int64, but got {binclf_curves.dtype}.")

    if (binclf_curves < 0).any():
        raise ValueError("Expected argument `binclf_curves` to have non-negative values, but got negative values.")

    neg = binclf_curves[:, :, 0, :].sum(dim=-1)  # (num_images, num_thresholds)
    if (neg != neg[:, 0].unsqueeze(1)).any():
        raise ValueError(
            "Expected argument `binclf_curves` to have the same number of negatives per image for every threshold."
        )

    pos = binclf_curves[:, :, 1, :].sum(dim=-1)  # (num_images, num_thresholds)
    if (pos != pos[:, 0].unsqueeze(1)).any():
        raise ValueError(
            "Expected argument `binclf_curves` to have the same number of positives per image for every threshold."
        )


# =========================================== FUNCTIONAL ===========================================


def __binclf_curves_ndarray_itertools(scores: ndarray, mask: ndarray, thresholds: ndarray):
    """Compute the binary classification matrix for a given sequence of thresholds.

    In the case where the thresholds are given (i.e. not considering all possible thresholds based on the scores),
    this weird-looking function is faster than the two options in `torchmetrics` on the CPU:
        - `_binary_precision_recall_curve_update_vectorized`
        - `_binary_precision_recall_curve_update_loop`

    (both in module `torchmetrics.functional.classification.precision_recall_curve` in `torchmetrics==1.1.0`).

    Notice the double underscore in the function name.
    The single underscore is reserved for the vectorized version (numpy.vectorize).

    ATTENTION:
        1. `thresholds` must be sorted in ascending order!
        2. Argument validation is not done here (only shapes)!

    Note: predicted as positive condition is `score >= th`.

    Args:
        scores (ndarray): Anomaly score maps of shape (N,),
        mask (ndarray): Binary (bool) ground truth mask of shape (N,),
        thresholds (ndarray): Sequence of thresholds to compute the binary classification matrix for.

    Returns:
        ndarray: Binary classification matrix of shape (num_thresholds, 2, 2)
        The last two dimensions are the confusion matrix for each threshold, organized as:
            - `tps`: `[... , 1, 1]`
            - `fps`: `[... , 0, 1]`
            - `fns`: `[... , 1, 0]`
            - `tns`: `[... , 0, 0]`
    """

    if scores.ndim != 1:
        raise ValueError(f"Expected argument `scores` to be 1D, but got {scores.ndim}")

    if mask.shape != scores.shape:
        raise ValueError(
            "Expected arguments `scores` and `mask` to have the same shape, "
            f"but got {scores.shape} and {mask.shape}."
        )

    if thresholds.ndim != 1:
        raise ValueError(f"Expected argument `thresholds` to be 1D, but got {thresholds.ndim}")

    num_th = len(thresholds)

    # POSITIVES
    scores_positives = scores[mask]
    # the sorting is very important for the algorithm to work and the speedup
    scores_positives = np.sort(scores_positives)
    # variable updated in the loop; start counting with lowest threshold ==> everything is predicted as positive
    num_pos = current_count_tp = scores_positives.size
    # `tp` stands for `true positive`
    tps = np.empty((num_th,), dtype=np.int64)

    # NEGATIVES
    # same thing but for the negative samples
    scores_negatives = scores[~mask]
    scores_negatives = np.sort(scores_negatives)
    num_neg = current_count_fp = scores_negatives.size
    # `fp` stands for `false positive`
    fps = np.empty((num_th,), dtype=np.int64)

    # it will progressively drop the scores that are below the current threshold
    for thidx, th in enumerate(thresholds):
        # UPDATE POSITIVES
        # < becasue it is the same as ~(>=)
        num_drop = sum(1 for _ in itertools.takewhile(lambda x: x < th, scores_positives))
        scores_positives = scores_positives[num_drop:]
        current_count_tp -= num_drop
        tps[thidx] = current_count_tp

        # UPDATE NEGATIVES
        # same with the negatives
        num_drop = sum(1 for _ in itertools.takewhile(lambda x: x < th, scores_negatives))
        scores_negatives = scores_negatives[num_drop:]
        current_count_fp -= num_drop
        fps[thidx] = current_count_fp

    # deduce the rest of the matrix counts
    # `fn` stands for `false negative
    fns = num_pos * np.ones((num_th,), dtype=np.int64) - tps
    # `tn` stands for `true negative`
    tns = num_neg * np.ones((num_th,), dtype=np.int64) - fps

    # sequence of dimensions is (thresholds, true class, predicted class)
    # `tps`: `confmat[..., 1, 1]`
    # `fps`: `confmat[..., 0, 1]`
    # `fns`: `confmat[..., 1, 0]`
    # `tns`: `confmat[..., 0, 0]`
    return np.stack(
        [
            np.stack([tns, fps], axis=-1),
            np.stack([fns, tps], axis=-1),
        ],
        axis=-1,
    ).transpose(0, 2, 1)


# vectorized version of the function above (single underscore in the name)
_binclf_curves_ndarray_itertools = np.vectorize(
    __binclf_curves_ndarray_itertools,
    signature="(n),(n),(k)->(k,2,2)",
)


def _perimg_binclf_curve_compute_cpu(
    anomaly_maps: Tensor,
    masks: Tensor,
    threshold_bounds: Tensor | tuple[float, float],
    num_thresholds: int,
):
    """Compute the binary classification matrix for a range of thresholds.

    This function expects the inputs to be on the CPU.
    # TODO: verify best option on GPU

    Inspired on `binary_precision_recall_curve` and `_binary_clf_curve`,
    from `torchmetrics.functional.classification.precision_recall_curve`.
    Simplifications:
        1. omitted argument `ignore_index: Optional[int]`, feature not available
        2. omitted argument `validate_args: bool`, always True
        3. `thresholds` can only be `int`

    Args:
        anomaly_maps (Tensor): Anomaly score maps of shape (N, H, W)
        masks (Tensor): Binary ground truth masks of shape (N, H, W)
        threshold_bounds (Tensor | tuple[float, float]): Lower and upper bounds for the thresholds.
        num_thresholds (int): Number of thresholds to compute between `threshold_bounds`.

    Returns:
        (Tensor, Tensor[int64]):
            [0] Thresholds of shape (num_thresholds,) and dtype `anomaly_maps.dtype`.

            [1] Binary classification matrices of shape (N, num_thresholds, 2, 2)
            N: number of images/instances
            The last two dimensions are the confusion matrix for each threshold, organized as:
                - `tps`: `[... , 1, 1]`
                - `fps`: `[... , 0, 1]`
                - `fns`: `[... , 1, 0]`
                - `tns`: `[... , 0, 0]`
    """
    # *** validate() ***
    _validate_num_thresholds(num_thresholds)
    th_lbound, th_ubound = threshold_bounds = torch.as_tensor(
        threshold_bounds, dtype=anomaly_maps.dtype, device=anomaly_maps.device
    )
    _validate_threshold_bounds(threshold_bounds)
    _validate_anomaly_maps_and_masks(anomaly_maps, masks)
    _validate_tensor_in_cpu(anomaly_maps)
    _validate_tensor_in_cpu(masks)

    # *** format() ***
    # `flatten(1)` will keep the batch dimension and flatten the rest
    # dim=0 is the batch dimension
    anomaly_maps = anomaly_maps.flatten(1)
    masks = masks.flatten(1)
    thresholds = torch.linspace(
        start=th_lbound,
        end=th_ubound,
        steps=num_thresholds,
        device=anomaly_maps.device,
        dtype=anomaly_maps.dtype,
    )

    # *** update() ***
    binclf_curve_ndarray = _binclf_curves_ndarray_itertools(
        anomaly_maps.numpy(), masks.numpy().astype(bool), thresholds.numpy()
    )
    return thresholds, torch.from_numpy(binclf_curve_ndarray).to(anomaly_maps.device).long()


# =========================================== METRIC ===========================================


class PerImageBinClfCurve(Metric):
    """Compute the binary classification matrix for a range of thresholds.

    ATTENTION: only cpu-based computation is supported for now.
    GPU alternatives are being speed-tested.
    """

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False

    num_thresholds: Tensor
    threshold_bounds: Tensor

    anomaly_maps: list[Tensor]
    masks: list[Tensor]
    image_classes: list[Tensor]

    def __init__(
        self,
        num_thresholds: int = 10_000,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        warnings.warn(
            f"Metric `{self.__class__.__name__}` will save all targets and predictions in buffer."
            " For large datasets this may lead to large memory footprint."
        )

        _validate_num_thresholds(num_thresholds)
        self.register_buffer("num_thresholds", torch.tensor(num_thresholds))

        # deduced from the anomaly maps in `compute()`
        self.register_buffer("threshold_bounds", torch.empty(2, dtype=torch.float32, device=torch.device("cpu")))

        self.add_state("anomaly_maps", default=[], dist_reduce_fx="cat")  # pylint: disable=not-callable
        self.add_state("masks", default=[], dist_reduce_fx="cat")  # pylint: disable=not-callable
        self.add_state("image_classes", default=[], dist_reduce_fx="cat")  # pylint: disable=not-callable

    def update(self, anomaly_maps: Tensor, masks: Tensor) -> None:  # type: ignore
        """Update state with new values.

        Args:
            anomaly_maps (Tensor): predictions of the model (ndim >= 2)
            masks (Tensor): ground truth masks (ndim >= 2, binary)
        """
        _validate_anomaly_maps_and_masks(anomaly_maps, masks)
        self.anomaly_maps.append(anomaly_maps)
        self.masks.append(masks)
        self.image_classes.append(
            # an image is anomalous if it has at least one anomaly pixel
            (masks.flatten(1) == 1)
            .any(dim=1)
            .to(torch.int32)
        )

    @property
    def is_empty(self) -> bool:
        """Return True if the metric has not been updated yet."""
        return len(self.image_classes) == 0

    def compute(self) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
        (Tensor[float], Tensor[int64], Tensor[int64]):

            [0] Thresholds of shape (num_thresholds,) and dtype equals to the anomaly maps from update().

            [1] Binary classification matrices of shape (N, num_thresholds, 2, 2), dtype float64
                N: number of images/instances seen during update() calls
                The last two dimensions are the confusion matrix for each threshold, organized as:
                    - `tps`: `[... , 1, 1]`
                    - `fps`: `[... , 0, 1]`
                    - `fns`: `[... , 1, 0]`
                    - `tns`: `[... , 0, 0]`
            [2] Image classes of shape (N,), dtype int32
        """

        if self.is_empty:
            return (
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, 2, 2, dtype=torch.int64),
                torch.empty(0, dtype=torch.int32),
            )

        anomaly_maps = dim_zero_cat(self.anomaly_maps).detach()
        masks = dim_zero_cat(self.masks).detach()
        image_classes = dim_zero_cat(self.image_classes).detach()

        self.threshold_bounds = torch.tensor((anomaly_maps.min(), anomaly_maps.max()))

        # TODO decide method for GPU
        thresholds, binclf_curves = _perimg_binclf_curve_compute_cpu(
            anomaly_maps=anomaly_maps.cpu(),
            masks=masks.cpu(),
            threshold_bounds=self.threshold_bounds,
            num_thresholds=self.num_thresholds.item(),
        )

        return thresholds, binclf_curves, image_classes

    @staticmethod
    def tprs(binclf_curves: Tensor) -> Tensor:
        """True positive rates (TPR) for image for each threshold.

        TPR = TP / P = TP / (TP + FN)

        Args:
            binclf_curves (Tensor): shape (N, num_thresholds, 2, 2), dtype int64
                                    output of PerImageBinClfCurve.compute()

        Returns:
            Tensor: shape (N, num_thresholds), dtype float64
            N: number of images/instances seen during update() calls
        """
        _validated_binclf_curves(binclf_curves)

        # (num_images, num_thresholds)
        tps = binclf_curves[..., 1, 1]
        pos = binclf_curves[..., 1, :].sum(dim=-1)

        # tprs will be nan if pos == 0 (normal image), which is expected
        tprs = tps.to(torch.float64) / pos.to(torch.float64)

        # (num_images, num_thresholds)
        return tprs

    @staticmethod
    def fprs(binclf_curves: Tensor) -> Tensor:
        """False positive rates (TPR) for image for each threshold.

        FPR = FP / N = FP / (FP + TN)

        Args:
            binclf_curves (Tensor): shape (N, num_thresholds, 2, 2), dtype int64
                                    output of PerImageBinClfCurve.compute()

        Returns:
            Tensor: shape (N, num_thresholds), dtype float64
            N: number of images/instances seen during update() calls
        """
        _validated_binclf_curves(binclf_curves)

        # (num_images, num_thresholds)
        fps = binclf_curves[..., 0, 1]
        neg = binclf_curves[..., 0, :].sum(dim=-1)

        # it can be `nan` if an anomalous image is fully covered by the mask
        fprs = fps.to(torch.float64) / neg.to(torch.float64)

        # (num_images, num_thresholds)
        return fprs
