"""Per-Image Overlap curve (PIMO, pronounced pee-mo) and its area under the curve (AUPIMO).

This module implements torch interfaces to access the numpy code in `pimo_numpy.py`.
Check its docstring for more details.

Tensors are build with `torch.from_numpy` and so the returned tensors will share the same memory as the numpy arrays.

Validations will preferably happen in ndarray so the numpy code can be reused without torch,
so often times the Tensor arguments will be converted to ndarray and then validated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from . import _validate, pimo_numpy
from .binclf_curve_numpy import Algorithm as BinclfAlgorithm
from .pimo_numpy import SharedFPRMetric

# =========================================== ARGS VALIDATION ===========================================


# =========================================== RESULT OBJECT ===========================================


# TODO(jpcbertoldo): missing docstring for `PIMOResult`  # noqa: TD003
@dataclass
class PIMOResult:  # noqa: D101
    # metadata
    shared_fpr_metric: str

    # data
    threshs: Tensor = field(repr=False)
    shared_fpr: Tensor = field(repr=False)
    per_image_tprs: Tensor = field(repr=False)

    @property
    def num_threshs(self) -> int:
        """Number of thresholds."""
        return self.threshs.shape[0]

    @property
    def num_images(self) -> int:
        """Number of images."""
        return self.per_image_tprs.shape[0]

    @property
    def image_classes(self) -> Tensor:
        """Image classes (0: normal, 1: anomalous)."""
        return (self.per_image_tprs.flatten(1) == 1).any(dim=1).to(torch.int32)

    def thresh_at(self, fpr_level: float) -> tuple[int, float, float]:
        """Return the threshold at the given shared FPR.

        See `anomalib.utils.metrics.per_image.pimo_numpy.thresh_at_shared_fpr_level` for details.

        Args:
            fpr_level (float): shared FPR level

        Returns:
            tuple[int, float, float]:
                [0] index of the threshold
                [1] threshold
                [2] the actual shared FPR value at the returned threshold
        """
        return pimo_numpy.thresh_at_shared_fpr_level(
            self.threshs.numpy(),
            self.shared_fpr.numpy(),
            fpr_level,
        )


# TODO(jpcbertoldo): missing docstring for `AUPIMOResult`  # noqa: TD003
# TODO(jpcbertoldo): change `aucs` in the paper supp mat to `aupimos`  # noqa: TD003
@dataclass
class AUPIMOResult:  # noqa: D101
    # metadata
    shared_fpr_metric: str
    fpr_lower_bound: float
    fpr_upper_bound: float
    num_threshs: int

    # data
    thresh_lower_bound: float = field(repr=False)
    thresh_upper_bound: float = field(repr=False)
    aupimos: Tensor = field(repr=False)

    @property
    def num_images(self) -> int:
        """Number of images."""
        return self.aupimos.shape[0]

    @property
    def image_classes(self) -> Tensor:
        """Image classes (0: normal, 1: anomalous)."""
        # if an instance has `nan` aupimo it's because it's a normal image
        return self.aupimos.isnan().to(torch.int32)


# =========================================== FUNCTIONAL ===========================================


# TODO(jpcbertoldo): missing docstring for `pimo`  # noqa: TD003
def pimo(  # noqa: D103
    anomaly_maps: Tensor,
    masks: Tensor,
    num_threshs: int,
    binclf_algorithm: str = BinclfAlgorithm.NUMBA,
    shared_fpr_metric: str = SharedFPRMetric.MEAN_PERIMAGE_FPR,
) -> PIMOResult:
    _validate.is_tensor(anomaly_maps, argname="anomaly_maps")
    anomaly_maps_array = anomaly_maps.detach().cpu().numpy()

    _validate.is_tensor(masks, argname="masks")
    masks_array = masks.detach().cpu().numpy()

    # other validations are done in the numpy code
    threshs_array, shared_fpr_array, per_image_tprs_array, _ = pimo_numpy.pimo(
        anomaly_maps_array,
        masks_array,
        num_threshs,
        binclf_algorithm=binclf_algorithm,
        shared_fpr_metric=shared_fpr_metric,
    )
    # _ is `image_classes` -- not needed here because it's a property in the result object

    # tensors are build with `torch.from_numpy` and so the returned tensors
    # will share the same memory as the numpy arrays
    device = anomaly_maps.device
    # N: number of images, K: number of thresholds
    # shape => (K,)
    threshs = torch.from_numpy(threshs_array).to(device)
    # shape => (K,)
    shared_fpr = torch.from_numpy(shared_fpr_array).to(device)
    # shape => (N, K)
    per_image_tprs = torch.from_numpy(per_image_tprs_array).to(device)

    return PIMOResult(
        shared_fpr_metric=shared_fpr_metric,
        threshs=threshs,
        shared_fpr=shared_fpr,
        per_image_tprs=per_image_tprs,
    )


# TODO(jpcbertoldo): missing docstring for `aupimo`  # noqa: TD003
def aupimo(  # noqa: D103
    anomaly_maps: Tensor,
    masks: Tensor,
    num_threshs: int = 300_000,
    binclf_algorithm: str = BinclfAlgorithm.NUMBA,
    shared_fpr_metric: str = SharedFPRMetric.MEAN_PERIMAGE_FPR,
    fpr_bounds: tuple[float, float] = (1e-5, 1e-4),
    force: bool = False,
) -> tuple[PIMOResult, AUPIMOResult]:
    _validate.is_tensor(anomaly_maps, argname="anomaly_maps")
    anomaly_maps_array = anomaly_maps.detach().cpu().numpy()

    _validate.is_tensor(masks, argname="masks")
    masks_array = masks.detach().cpu().numpy()

    # other validations are done in the numpy code

    threshs_array, shared_fpr_array, per_image_tprs_array, _, aupimos_array = pimo_numpy.aupimo(
        anomaly_maps_array,
        masks_array,
        num_threshs,
        binclf_algorithm=binclf_algorithm,
        shared_fpr_metric=shared_fpr_metric,
        fpr_bounds=fpr_bounds,
        force=force,
    )

    # tensors are build with `torch.from_numpy` and so the returned tensors
    # will share the same memory as the numpy arrays
    device = anomaly_maps.device
    # N: number of images, K: number of thresholds
    # shape => (K,)
    threshs = torch.from_numpy(threshs_array).to(device)
    # shape => (K,)
    shared_fpr = torch.from_numpy(shared_fpr_array).to(device)
    # shape => (N, K)
    per_image_tprs = torch.from_numpy(per_image_tprs_array).to(device)
    # shape => (N,)
    aupimos = torch.from_numpy(aupimos_array).to(device)

    pimoresult = PIMOResult(
        shared_fpr_metric=shared_fpr_metric,
        threshs=threshs,
        shared_fpr=shared_fpr,
        per_image_tprs=per_image_tprs,
    )
    fpr_lower_bound, fpr_upper_bound = fpr_bounds
    # recall: fpr upper/lower bounds are the same as the thresh lower/upper bounds
    # `_` is the threshold's index, `__` is the actual fpr value
    _, thresh_lower_bound, __ = pimoresult.thresh_at(fpr_upper_bound)
    _, thresh_upper_bound, __ = pimoresult.thresh_at(fpr_lower_bound)
    return (
        pimoresult,
        AUPIMOResult(
            shared_fpr_metric=shared_fpr_metric,
            fpr_lower_bound=(fpr_lower_bound),
            fpr_upper_bound=(fpr_upper_bound),
            num_threshs=num_threshs,
            thresh_lower_bound=thresh_lower_bound,
            thresh_upper_bound=thresh_upper_bound,
            aupimos=aupimos,
        ),
    )
