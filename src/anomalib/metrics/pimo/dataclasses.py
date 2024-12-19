"""Dataclasses for PIMO metrics.

This module provides dataclasses for storing and manipulating PIMO (Per-Image
Metric Optimization) and AUPIMO (Area Under PIMO) results.

The dataclasses include:

- ``PIMOResult``: Container for PIMO curve data and metadata
- ``AUPIMOResult``: Container for AUPIMO curve data and metadata

Example:
    >>> from anomalib.metrics.pimo.dataclasses import PIMOResult
    >>> import torch
    >>> thresholds = torch.linspace(0, 1, 10)
    >>> shared_fpr = torch.linspace(1, 0, 10)  # Decreasing FPR
    >>> per_image_tprs = torch.rand(5, 10)  # 5 images, 10 thresholds
    >>> result = PIMOResult(thresholds, shared_fpr, per_image_tprs)
    >>> result.num_images
    5
"""

# Based on the code: https://github.com/jpcbertoldo/aupimo
#
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import torch

from . import _validate, functional


@dataclass
class PIMOResult:
    """Per-Image Overlap (PIMO, pronounced pee-mo) curve.

    This class stores PIMO curve data and metadata and provides utility methods
    for analysis.

    Notation:
        - ``N``: number of images
        - ``K``: number of thresholds
        - ``FPR``: False Positive Rate
        - ``TPR``: True Positive Rate

    Args:
        thresholds: Sequence of ``K`` monotonically increasing thresholds used
            to compute the PIMO curve. Shape: ``(K,)``
        shared_fpr: ``K`` values of the shared FPR metric at corresponding
            thresholds. Shape: ``(K,)``
        per_image_tprs: For each of the ``N`` images, the ``K`` values of
            in-image TPR at corresponding thresholds. Shape: ``(N, K)``

    Example:
        >>> import torch
        >>> thresholds = torch.linspace(0, 1, 10)
        >>> shared_fpr = torch.linspace(1, 0, 10)  # Decreasing FPR
        >>> per_image_tprs = torch.rand(5, 10)  # 5 images, 10 thresholds
        >>> result = PIMOResult(thresholds, shared_fpr, per_image_tprs)
        >>> result.num_images
        5
    """

    # data
    thresholds: torch.Tensor = field(repr=False)  # shape => (K,)
    shared_fpr: torch.Tensor = field(repr=False)  # shape => (K,)
    per_image_tprs: torch.Tensor = field(repr=False)  # shape => (N, K)

    @property
    def num_threshsholds(self) -> int:
        """Get number of thresholds.

        Returns:
            Number of thresholds used in the PIMO curve.
        """
        return self.thresholds.shape[0]

    @property
    def num_images(self) -> int:
        """Get number of images.

        Returns:
            Number of images in the dataset.
        """
        return self.per_image_tprs.shape[0]

    @property
    def image_classes(self) -> torch.Tensor:
        """Get image classes (0: normal, 1: anomalous).

        The class is deduced from the per-image TPRs. If any TPR value is not
        NaN, the image is considered anomalous.

        Returns:
            Tensor of shape ``(N,)`` containing image classes.
        """
        return (~torch.isnan(self.per_image_tprs)).any(dim=1).to(torch.int32)

    def __post_init__(self) -> None:
        """Validate inputs for result object consistency.

        Raises:
            TypeError: If inputs are invalid or have inconsistent shapes.
        """
        try:
            _validate.is_valid_threshold(self.thresholds)
            _validate.is_rate_curve(self.shared_fpr, nan_allowed=False, decreasing=True)  # is_shared_apr
            _validate.is_per_image_tprs(self.per_image_tprs, self.image_classes)

        except (TypeError, ValueError) as ex:
            msg = f"Invalid inputs for {self.__class__.__name__} object. Cause: {ex}."
            raise TypeError(msg) from ex

        if self.thresholds.shape != self.shared_fpr.shape:
            msg = (
                f"Invalid {self.__class__.__name__} object. "
                f"Attributes have inconsistent shapes: "
                f"{self.thresholds.shape=} != {self.shared_fpr.shape=}."
            )
            raise TypeError(msg)

        if self.thresholds.shape[0] != self.per_image_tprs.shape[1]:
            msg = (
                f"Invalid {self.__class__.__name__} object. "
                f"Attributes have inconsistent shapes: "
                f"{self.thresholds.shape[0]=} != {self.per_image_tprs.shape[1]=}."
            )
            raise TypeError(msg)

    def thresh_at(self, fpr_level: float) -> tuple[int, float, float]:
        """Get threshold at given shared FPR level.

        For details see
        :func:`anomalib.metrics.per_image.pimo_numpy.thresh_at_shared_fpr_level`.

        Args:
            fpr_level: Target shared FPR level to find threshold for.

        Returns:
            Tuple containing:
                - Index of the threshold
                - Threshold value
                - Actual shared FPR value at returned threshold

        Example:
            >>> result = PIMOResult(...)  # doctest: +SKIP
            >>> idx, thresh, fpr = result.thresh_at(0.1)  # doctest: +SKIP
        """
        idx, thresh, fpr = functional.thresh_at_shared_fpr_level(
            self.thresholds,
            self.shared_fpr,
            fpr_level,
        )
        return idx, thresh, float(fpr)


@dataclass
class AUPIMOResult:
    """Area Under Per-Image Overlap (AUPIMO, pronounced a-u-pee-mo) curve.

    This class stores AUPIMO data and metadata and provides utility methods for
    analysis.

    Args:
        fpr_lower_bound: Lower bound of the FPR integration range.
        fpr_upper_bound: Upper bound of the FPR integration range.
        num_thresholds: Number of thresholds used to compute AUPIMO. Note this
            is different from thresholds used for PIMO curve.
        thresh_lower_bound: Lower threshold bound (corresponds to upper FPR).
        thresh_upper_bound: Upper threshold bound (corresponds to lower FPR).
        aupimos: AUPIMO scores, one per image. Shape: ``(N,)``

    Example:
        >>> import torch
        >>> aupimos = torch.rand(5)  # 5 images
        >>> result = AUPIMOResult(  # doctest: +SKIP
        ...     fpr_lower_bound=0.0,
        ...     fpr_upper_bound=0.3,
        ...     num_thresholds=100,
        ...     thresh_lower_bound=0.5,
        ...     thresh_upper_bound=0.9,
        ...     aupimos=aupimos
        ... )
    """

    # metadata
    fpr_lower_bound: float
    fpr_upper_bound: float
    num_thresholds: int | None

    # data
    thresh_lower_bound: float = field(repr=False)
    thresh_upper_bound: float = field(repr=False)
    aupimos: torch.Tensor = field(repr=False)  # shape => (N,)

    @property
    def num_images(self) -> int:
        """Get number of images.

        Returns:
            Number of images in dataset.
        """
        return self.aupimos.shape[0]

    @property
    def num_normal_images(self) -> int:
        """Get number of normal images.

        Returns:
            Count of images with class 0 (normal).
        """
        return int((self.image_classes == 0).sum())

    @property
    def num_anomalous_images(self) -> int:
        """Get number of anomalous images.

        Returns:
            Count of images with class 1 (anomalous).
        """
        return int((self.image_classes == 1).sum())

    @property
    def image_classes(self) -> torch.Tensor:
        """Get image classes (0: normal, 1: anomalous).

        An image is considered normal if its AUPIMO score is NaN.

        Returns:
            Tensor of shape ``(N,)`` containing image classes.
        """
        return self.aupimos.isnan().to(torch.int32)

    @property
    def fpr_bounds(self) -> tuple[float, float]:
        """Get FPR integration range bounds.

        Returns:
            Tuple of (lower bound, upper bound) for FPR range.
        """
        return self.fpr_lower_bound, self.fpr_upper_bound

    @property
    def thresh_bounds(self) -> tuple[float, float]:
        """Get threshold integration range bounds.

        Note:
            Bounds correspond to FPR bounds in reverse order:
                - ``fpr_lower_bound`` -> ``thresh_upper_bound``
                - ``fpr_upper_bound`` -> ``thresh_lower_bound``

        Returns:
            Tuple of (lower bound, upper bound) for threshold range.
        """
        return self.thresh_lower_bound, self.thresh_upper_bound

    def __post_init__(self) -> None:
        """Validate inputs for result object consistency.

        Raises:
            TypeError: If inputs are invalid.
        """
        try:
            _validate.is_rate_range(
                (self.fpr_lower_bound, self.fpr_upper_bound),
            )
            # TODO(jpcbertoldo): warn when too low (use numpy code params)  # noqa: TD003
            if self.num_thresholds is not None:
                _validate.is_num_thresholds_gte2(self.num_thresholds)
            _validate.is_rates(self.aupimos, nan_allowed=True)

            _validate.validate_threshold_bounds(
                (self.thresh_lower_bound, self.thresh_upper_bound),
            )

        except (TypeError, ValueError) as ex:
            msg = f"Invalid inputs for {self.__class__.__name__} object. Cause: {ex}."
            raise TypeError(msg) from ex

    @classmethod
    def from_pimo_result(
        cls: type["AUPIMOResult"],
        pimo_result: PIMOResult,
        fpr_bounds: tuple[float, float],
        num_thresholds_auc: int,
        aupimos: torch.Tensor,
    ) -> "AUPIMOResult":
        """Create AUPIMO result from PIMO result.

        Args:
            pimo_result: Source PIMO result object.
            fpr_bounds: Tuple of (lower, upper) bounds for FPR range.
            num_thresholds_auc: Number of thresholds for AUPIMO computation.
                Note this differs from PIMO curve thresholds.
            aupimos: AUPIMO scores, one per image.

        Returns:
            New AUPIMO result object.

        Raises:
            TypeError: If inputs are invalid or inconsistent.

        Example:
            >>> pimo_result = PIMOResult(...)  # doctest: +SKIP
            >>> aupimos = torch.rand(5)  # 5 images
            >>> result = AUPIMOResult.from_pimo_result(  # doctest: +SKIP
            ...     pimo_result=pimo_result,
            ...     fpr_bounds=(0.0, 0.3),
            ...     num_thresholds_auc=100,
            ...     aupimos=aupimos
            ... )
        """
        if pimo_result.per_image_tprs.shape[0] != aupimos.shape[0]:
            msg = (
                f"Invalid {cls.__name__} object. "
                f"Attributes have inconsistent shapes: "
                f"there are {pimo_result.per_image_tprs.shape[0]} PIMO curves "
                f"but {aupimos.shape[0]} AUPIMO scores."
            )
            raise TypeError(msg)

        if not torch.isnan(aupimos[pimo_result.image_classes == 0]).all():
            msg = "Expected all normal images to have NaN AUPIMOs, but some have non-NaN values."
            raise TypeError(msg)

        if torch.isnan(aupimos[pimo_result.image_classes == 1]).any():
            msg = "Expected all anomalous images to have valid AUPIMOs (not nan), but some have NaN values."
            raise TypeError(msg)

        fpr_lower_bound, fpr_upper_bound = fpr_bounds
        # recall: fpr upper/lower bounds are same as thresh lower/upper bounds
        _, thresh_lower_bound, __ = pimo_result.thresh_at(fpr_upper_bound)
        _, thresh_upper_bound, __ = pimo_result.thresh_at(fpr_lower_bound)
        # `_` is threshold's index, `__` is actual fpr value
        return cls(
            fpr_lower_bound=fpr_lower_bound,
            fpr_upper_bound=fpr_upper_bound,
            num_thresholds=num_thresholds_auc,
            thresh_lower_bound=float(thresh_lower_bound),
            thresh_upper_bound=float(thresh_upper_bound),
            aupimos=aupimos,
        )
