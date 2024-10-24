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

# Disclaimer

This module implements torch interfaces to access the numpy code in `pimo_numpy.py`.
Tensors are converted to numpy arrays and then passed and validated in the numpy code.
The results are converted back to tensors and eventually wrapped in an dataclass object.

Validations will preferably happen in ndarray so the numpy code can be reused without torch,
so often times the Tensor arguments will be converted to ndarray and then validated.
"""

# Original Code
# https://github.com/jpcbertoldo/aupimo
#
# Modified
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import torch
from torchmetrics import Metric

from . import _validate, functional
from .dataclasses import AUPIMOResult, PIMOResult

logger = logging.getLogger(__name__)


class PIMO(Metric):
    """Per-IMage Overlap (PIMO, pronounced pee-mo) curves.

    This torchmetrics interface is a wrapper around the functional interface, which is a wrapper around the numpy code.
    The tensors are converted to numpy arrays and then passed and validated in the numpy code.
    The results are converted back to tensors and wrapped in an dataclass object.

    PIMO is a curve of True Positive Rate (TPR) values on each image across multiple anomaly score thresholds.
    The anomaly score thresholds are indexed by a (cross-image shared) value of False Positive Rate (FPR) measure on
    the normal images.

    Details: `anomalib.metrics.per_image.pimo`.

    Notation:
        N: number of images
        H: image height
        W: image width
        K: number of thresholds

    Attributes:
        anomaly_maps: floating point anomaly score maps of shape (N, H, W)
        masks: binary (bool or int) ground truth masks of shape (N, H, W)

    Args:
        num_thresholds: number of thresholds to compute (K)
        binclf_algorithm: algorithm to compute the binary classifier curve (see `binclf_curve_numpy.Algorithm`)

    Returns:
        PIMOResult: PIMO curves dataclass object. See `PIMOResult` for details.
    """

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False

    num_thresholds: int
    binclf_algorithm: str

    anomaly_maps: list[torch.Tensor]
    masks: list[torch.Tensor]

    @property
    def _is_empty(self) -> bool:
        """Return True if the metric has not been updated yet."""
        return len(self.anomaly_maps) == 0

    @property
    def num_images(self) -> int:
        """Number of images."""
        return sum(am.shape[0] for am in self.anomaly_maps)

    @property
    def image_classes(self) -> torch.Tensor:
        """Image classes (0: normal, 1: anomalous)."""
        return functional.images_classes_from_masks(self.masks)

    def __init__(self, num_thresholds: int) -> None:
        """Per-Image Overlap (PIMO) curve.

        Args:
            num_thresholds: number of thresholds used to compute the PIMO curve (K)
        """
        super().__init__()

        logger.warning(
            f"Metric `{self.__class__.__name__}` will save all targets and predictions in buffer."
            " For large datasets this may lead to large memory footprint.",
        )

        # the options below are, redundantly, validated here to avoid reaching
        # an error later in the execution

        _validate.is_num_thresholds_gte2(num_thresholds)
        self.num_thresholds = num_thresholds

        self.add_state("anomaly_maps", default=[], dist_reduce_fx="cat")
        self.add_state("masks", default=[], dist_reduce_fx="cat")

    def update(self, anomaly_maps: torch.Tensor, masks: torch.Tensor) -> None:
        """Update lists of anomaly maps and masks.

        Args:
            anomaly_maps (torch.Tensor): predictions of the model (ndim == 2, float)
            masks (torch.Tensor): ground truth masks (ndim == 2, binary)
        """
        _validate.is_anomaly_maps(anomaly_maps)
        _validate.is_masks(masks)
        _validate.is_same_shape(anomaly_maps, masks)
        self.anomaly_maps.append(anomaly_maps)
        self.masks.append(masks)

    def compute(self) -> PIMOResult:
        """Compute the PIMO curves.

        Call the functional interface `pimo_curves()`, which is a wrapper around the numpy code.

        Returns:
            PIMOResult: PIMO curves dataclass object. See `PIMOResult` for details.
        """
        if self._is_empty:
            msg = "No anomaly maps and masks have been added yet. Please call `update()` first."
            raise RuntimeError(msg)
        anomaly_maps = torch.concat(self.anomaly_maps, dim=0)
        masks = torch.concat(self.masks, dim=0)

        thresholds, shared_fpr, per_image_tprs, _ = functional.pimo_curves(
            anomaly_maps,
            masks,
            self.num_thresholds,
        )
        return PIMOResult(
            thresholds=thresholds,
            shared_fpr=shared_fpr,
            per_image_tprs=per_image_tprs,
        )


class AUPIMO(PIMO):
    """Area Under the Per-Image Overlap (PIMO) curve.

    This torchmetrics interface is a wrapper around the functional interface, which is a wrapper around the numpy code.
    The tensors are converted to numpy arrays and then passed and validated in the numpy code.
    The results are converted back to tensors and wrapped in an dataclass object.

    Scores are computed from the integration of the PIMO curves within the given FPR bounds, then normalized to [0, 1].
    It can be thought of as the average TPR of the PIMO curves within the given FPR bounds.

    Details: `anomalib.metrics.per_image.pimo`.

    Notation:
        N: number of images
        H: image height
        W: image width
        K: number of thresholds

    Attributes:
        anomaly_maps: floating point anomaly score maps of shape (N, H, W)
        masks: binary (bool or int) ground truth masks of shape (N, H, W)

    Args:
        num_thresholds: number of thresholds to compute (K)
        fpr_bounds: lower and upper bounds of the FPR integration range
        force: whether to force the computation despite bad conditions

    Returns:
        tuple[PIMOResult, AUPIMOResult]: PIMO and AUPIMO results dataclass objects. See `PIMOResult` and `AUPIMOResult`.
    """

    fpr_bounds: tuple[float, float]
    return_average: bool
    force: bool

    @staticmethod
    def normalizing_factor(fpr_bounds: tuple[float, float]) -> float:
        """Constant that normalizes the AUPIMO integral to 0-1 range.

        It is the maximum possible value from the integral in AUPIMO's definition.
        It corresponds to assuming a constant function T_i: thresh --> 1.

        Args:
            fpr_bounds: lower and upper bounds of the FPR integration range.

        Returns:
            float: the normalization factor (>0).
        """
        return functional.aupimo_normalizing_factor(fpr_bounds)

    def __repr__(self) -> str:
        """Show the metric name and its integration bounds."""
        lower, upper = self.fpr_bounds
        return f"{self.__class__.__name__}([{lower:.2g}, {upper:.2g}])"

    def __init__(
        self,
        num_thresholds: int = 300_000,
        fpr_bounds: tuple[float, float] = (1e-5, 1e-4),
        return_average: bool = True,
        force: bool = False,
    ) -> None:
        """Area Under the Per-Image Overlap (PIMO) curve.

        Args:
            num_thresholds: [passed to parent `PIMO`] number of thresholds used to compute the PIMO curve
            fpr_bounds: lower and upper bounds of the FPR integration range
            return_average: if True, return the average AUPIMO score; if False, return all the individual AUPIMO scores
            force: if True, force the computation of the AUPIMO scores even in bad conditions (e.g. few points)
        """
        super().__init__(num_thresholds=num_thresholds)

        # other validations are done in PIMO.__init__()

        _validate.is_rate_range(fpr_bounds)
        self.fpr_bounds = fpr_bounds
        self.return_average = return_average
        self.force = force

    def compute(self, force: bool | None = None) -> tuple[PIMOResult, AUPIMOResult]:  # type: ignore[override]
        """Compute the PIMO curves and their Area Under the curve (AUPIMO) scores.

        Call the functional interface `aupimo_scores()`, which is a wrapper around the numpy code.

        Args:
            force: if given (not None), override the `force` attribute.

        Returns:
            tuple[PIMOResult, AUPIMOResult]: PIMO curves and AUPIMO scores dataclass objects.
                See `PIMOResult` and `AUPIMOResult` for details.
        """
        if self._is_empty:
            msg = "No anomaly maps and masks have been added yet. Please call `update()` first."
            raise RuntimeError(msg)
        anomaly_maps = torch.concat(self.anomaly_maps, dim=0)
        masks = torch.concat(self.masks, dim=0)
        force = force if force is not None else self.force

        # other validations are done in the numpy code

        thresholds, shared_fpr, per_image_tprs, _, aupimos, num_thresholds_auc = functional.aupimo_scores(
            anomaly_maps,
            masks,
            self.num_thresholds,
            fpr_bounds=self.fpr_bounds,
            force=force,
        )

        pimo_result = PIMOResult(
            thresholds=thresholds,
            shared_fpr=shared_fpr,
            per_image_tprs=per_image_tprs,
        )
        aupimo_result = AUPIMOResult.from_pimo_result(
            pimo_result,
            fpr_bounds=self.fpr_bounds,
            # not `num_thresholds`!
            # `num_thresholds` is the number of thresholds used to compute the PIMO curve
            # this is the number of thresholds used to compute the AUPIMO integral
            num_thresholds_auc=num_thresholds_auc,
            aupimos=aupimos,
        )
        if self.return_average:
            # normal images have NaN AUPIMO scores
            is_nan = torch.isnan(aupimo_result.aupimos)
            return aupimo_result.aupimos[~is_nan].mean()
        return pimo_result, aupimo_result
