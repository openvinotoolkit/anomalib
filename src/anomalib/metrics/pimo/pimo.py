"""Per-Image Overlap curve (PIMO) and its area under the curve (AUPIMO).

This module provides metrics for evaluating anomaly detection performance using
Per-Image Overlap (PIMO) curves and their area under the curve (AUPIMO).

PIMO Curves
----------
PIMO curves plot True Positive Rate (TPR) values for each image across multiple
anomaly score thresholds. The thresholds are indexed by a shared False Positive
Rate (FPR) measure computed on normal images.

Each anomalous image has its own curve with:

- X-axis: Shared FPR (logarithmic average of per-image FPR on normal images)
- Y-axis: Per-image TPR ("Overlap" between ground truth and predicted masks)

Note on Shared FPR
----------------
The shared FPR metric can be made stricter by using cross-image max or high
percentile FPRs instead of mean. This further penalizes models with exceptional
false positives in normal images. Currently only mean FPR is implemented.

AUPIMO Score
-----------
AUPIMO is the area under each PIMO curve within bounded FPR integration range.
The score is normalized to [0,1].

Implementation Notes
------------------
This module implements PyTorch interfaces to the numpy implementation in
``pimo_numpy.py``. Tensors are converted to numpy arrays for computation and
validation, then converted back to tensors and wrapped in dataclass objects.

Example:
    >>> import torch
    >>> from anomalib.metrics.pimo import PIMO
    >>> metric = PIMO(num_thresholds=10)
    >>> anomaly_maps = torch.rand(5, 32, 32)  # 5 images
    >>> masks = torch.randint(0, 2, (5, 32, 32))  # Binary masks
    >>> metric.update(anomaly_maps, masks)
    >>> result = metric.compute()
    >>> result.num_images
    5

See Also:
    - :class:`PIMOResult`: Container for PIMO curve data
    - :class:`AUPIMOResult`: Container for AUPIMO score data
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

from anomalib.metrics.base import AnomalibMetric

from . import _validate, functional
from .dataclasses import AUPIMOResult, PIMOResult

logger = logging.getLogger(__name__)


class _PIMO(Metric):
    """Per-Image Overlap (PIMO) curve metric.

    This metric computes PIMO curves which plot True Positive Rate (TPR) values
    for each image across multiple anomaly score thresholds. The thresholds are
    indexed by a shared False Positive Rate (FPR) measure on normal images.

    Args:
        num_thresholds: Number of thresholds to compute (K). Must be >= 2.

    Attributes:
        anomaly_maps: List of anomaly score maps, each of shape ``(N, H, W)``
        masks: List of binary ground truth masks, each of shape ``(N, H, W)``
        is_differentiable: Whether metric is differentiable
        higher_is_better: Whether higher values are better
        full_state_update: Whether to update full state

    Example:
        >>> import torch
        >>> metric = _PIMO(num_thresholds=10)
        >>> anomaly_maps = torch.rand(5, 32, 32)  # 5 images
        >>> masks = torch.randint(0, 2, (5, 32, 32))  # Binary masks
        >>> metric.update(anomaly_maps, masks)
        >>> result = metric.compute()
        >>> result.num_images
        5

    Note:
        This metric stores all predictions and targets in memory, which may
        require significant memory for large datasets.
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
        """Check if metric has been updated.

        Returns:
            bool: True if no updates have been made yet.
        """
        return len(self.anomaly_maps) == 0

    @property
    def num_images(self) -> int:
        """Get total number of images.

        Returns:
            int: Total number of images across all batches.
        """
        return sum(am.shape[0] for am in self.anomaly_maps)

    @property
    def image_classes(self) -> torch.Tensor:
        """Get image classes (0: normal, 1: anomalous).

        Returns:
            torch.Tensor: Binary tensor of image classes.
        """
        return functional.images_classes_from_masks(self.masks)

    def __init__(self, num_thresholds: int) -> None:
        """Initialize PIMO metric.

        Args:
            num_thresholds: Number of thresholds for curve computation (K).
                Must be >= 2.
        """
        super().__init__()

        logger.warning(
            f"Metric `{self.__class__.__name__}` will save all targets and "
            "predictions in buffer. For large datasets this may lead to large "
            "memory footprint.",
        )

        # Validate options early to avoid later errors
        _validate.is_num_thresholds_gte2(num_thresholds)
        self.num_thresholds = num_thresholds

        self.add_state("anomaly_maps", default=[], dist_reduce_fx="cat")
        self.add_state("masks", default=[], dist_reduce_fx="cat")

    def update(self, anomaly_maps: torch.Tensor, masks: torch.Tensor) -> None:
        """Update metric state with new predictions and targets.

        Args:
            anomaly_maps: Model predictions as float tensors of shape
                ``(N, H, W)``
            masks: Ground truth binary masks of shape ``(N, H, W)``

        Raises:
            ValueError: If inputs have invalid shapes or types
        """
        _validate.is_anomaly_maps(anomaly_maps)
        _validate.is_masks(masks)
        _validate.is_same_shape(anomaly_maps, masks)
        self.anomaly_maps.append(anomaly_maps)
        self.masks.append(masks)

    def compute(self) -> PIMOResult:
        """Compute PIMO curves from accumulated data.

        Returns:
            PIMOResult: Container with curve data and metadata.

        Raises:
            RuntimeError: If no data has been added via update()
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


class PIMO(AnomalibMetric, _PIMO):  # type: ignore[misc]
    """Wrapper adding AnomalibMetric functionality to PIMO metric."""

    default_fields = ("anomaly_map", "gt_mask")


class _AUPIMO(_PIMO):
    """Area Under the Per-Image Overlap (PIMO) curve.

    This metric computes both PIMO curves and their area under the curve
    (AUPIMO). AUPIMO scores are computed by integrating PIMO curves within
    specified FPR bounds and normalizing to [0,1].

    Args:
        num_thresholds: Number of thresholds for curve computation. Default:
            300,000
        fpr_bounds: Lower and upper FPR integration bounds as ``(min, max)``.
            Default: ``(1e-5, 1e-4)``
        return_average: If True, return mean AUPIMO score across anomalous
            images. If False, return individual scores. Default: True
        force: If True, compute scores even in suboptimal conditions.
            Default: False

    Example:
        >>> import torch
        >>> metric = _AUPIMO(num_thresholds=10)
        >>> anomaly_maps = torch.rand(5, 32, 32)  # 5 images
        >>> masks = torch.randint(0, 2, (5, 32, 32))  # Binary masks
        >>> metric.update(anomaly_maps, masks)
        >>> pimo_result, aupimo_result = metric.compute()
        >>> aupimo_result.num_images
        5
    """

    fpr_bounds: tuple[float, float]
    return_average: bool
    force: bool

    @staticmethod
    def normalizing_factor(fpr_bounds: tuple[float, float]) -> float:
        """Get normalization factor for AUPIMO integral.

        The factor normalizes the integral to [0,1] range. It represents the
        maximum possible integral value, assuming a constant TPR of 1.

        Args:
            fpr_bounds: FPR integration bounds as ``(min, max)``

        Returns:
            float: Normalization factor (>0)
        """
        return functional.aupimo_normalizing_factor(fpr_bounds)

    def __repr__(self) -> str:
        """Get string representation with integration bounds.

        Returns:
            str: Metric name and FPR bounds
        """
        lower, upper = self.fpr_bounds
        return f"{self.__class__.__name__}([{lower:.2g}, {upper:.2g}])"

    def __init__(
        self,
        num_thresholds: int = 300_000,
        fpr_bounds: tuple[float, float] = (1e-5, 1e-4),
        return_average: bool = True,
        force: bool = False,
    ) -> None:
        """Initialize AUPIMO metric.

        Args:
            num_thresholds: Number of thresholds for curve computation
            fpr_bounds: FPR integration bounds as ``(min, max)``
            return_average: If True, return mean score across anomalous images
            force: If True, compute scores even in suboptimal conditions
        """
        super().__init__(num_thresholds=num_thresholds)

        _validate.is_rate_range(fpr_bounds)
        self.fpr_bounds = fpr_bounds
        self.return_average = return_average
        self.force = force

    def compute(self, force: bool | None = None) -> tuple[PIMOResult, AUPIMOResult]:  # type: ignore[override]
        """Compute PIMO curves and AUPIMO scores.

        Args:
            force: If provided, override instance ``force`` setting

        Returns:
            tuple: Contains:
                - PIMOResult: PIMO curve data
                - AUPIMOResult: AUPIMO score data

        Raises:
            RuntimeError: If no data has been added via update()
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


class AUPIMO(AnomalibMetric, _AUPIMO):  # type: ignore[misc]
    """Wrapper adding AnomalibMetric functionality to AUPIMO metric."""

    default_fields = ("anomaly_map", "gt_mask")
