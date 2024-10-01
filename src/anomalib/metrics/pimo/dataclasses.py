"""Dataclasses for PIMO metrics."""

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

    This interface gathers the PIMO curve data and metadata and provides several utility methods.

    Notation:
        - N: number of images
        - K: number of thresholds
        - FPR: False Positive Rate
        - TPR: True Positive Rate

    Attributes:
        thresholds (torch.Tensor): sequence of K (monotonically increasing) thresholds used to compute the PIMO curve
        shared_fpr (torch.Tensor): K values of the shared FPR metric at the corresponding thresholds
        per_image_tprs (torch.Tensor): for each of the N images, the K values of in-image TPR at the corresponding
            thresholds
    """

    # data
    thresholds: torch.Tensor = field(repr=False)  # shape => (K,)
    shared_fpr: torch.Tensor = field(repr=False)  # shape => (K,)
    per_image_tprs: torch.Tensor = field(repr=False)  # shape => (N, K)

    @property
    def num_threshsholds(self) -> int:
        """Number of thresholds."""
        return self.thresholds.shape[0]

    @property
    def num_images(self) -> int:
        """Number of images."""
        return self.per_image_tprs.shape[0]

    @property
    def image_classes(self) -> torch.Tensor:
        """Image classes (0: normal, 1: anomalous).

        Deduced from the per-image TPRs.
        If any TPR value is not NaN, the image is considered anomalous.
        """
        return (~torch.isnan(self.per_image_tprs)).any(dim=1).to(torch.int32)

    def __post_init__(self) -> None:
        """Validate the inputs for the result object are consistent."""
        try:
            _validate.is_valid_threshold(self.thresholds)
            _validate.is_rate_curve(self.shared_fpr, nan_allowed=False, decreasing=True)  # is_shared_apr
            _validate.is_per_image_tprs(self.per_image_tprs, self.image_classes)

        except (TypeError, ValueError) as ex:
            msg = f"Invalid inputs for {self.__class__.__name__} object. Cause: {ex}."
            raise TypeError(msg) from ex

        if self.thresholds.shape != self.shared_fpr.shape:
            msg = (
                f"Invalid {self.__class__.__name__} object. Attributes have inconsistent shapes: "
                f"{self.thresholds.shape=} != {self.shared_fpr.shape=}."
            )
            raise TypeError(msg)

        if self.thresholds.shape[0] != self.per_image_tprs.shape[1]:
            msg = (
                f"Invalid {self.__class__.__name__} object. Attributes have inconsistent shapes: "
                f"{self.thresholds.shape[0]=} != {self.per_image_tprs.shape[1]=}."
            )
            raise TypeError(msg)

    def thresh_at(self, fpr_level: float) -> tuple[int, float, float]:
        """Return the threshold at the given shared FPR.

        See `anomalib.metrics.per_image.pimo_numpy.thresh_at_shared_fpr_level` for details.

        Args:
            fpr_level (float): shared FPR level

        Returns:
            tuple[int, float, float]:
                [0] index of the threshold
                [1] threshold
                [2] the actual shared FPR value at the returned threshold
        """
        return functional.thresh_at_shared_fpr_level(
            self.thresholds,
            self.shared_fpr,
            fpr_level,
        )


@dataclass
class AUPIMOResult:
    """Area Under the Per-Image Overlap (AUPIMO, pronounced a-u-pee-mo) curve.

    This interface gathers the AUPIMO data and metadata and provides several utility methods.

    Attributes:
        fpr_lower_bound (float): [metadata] LOWER bound of the FPR integration range
        fpr_upper_bound (float): [metadata] UPPER bound of the FPR integration range
        num_thresholds (int): [metadata] number of thresholds used to effectively compute AUPIMO;
                            should not be confused with the number of thresholds used to compute the PIMO curve
        thresh_lower_bound (float): LOWER threshold bound --> corresponds to the UPPER FPR bound
        thresh_upper_bound (float): UPPER threshold bound --> corresponds to the LOWER FPR bound
        aupimos (torch.Tensor): values of AUPIMO scores (1 per image)
    """

    # metadata
    fpr_lower_bound: float
    fpr_upper_bound: float
    num_thresholds: int

    # data
    thresh_lower_bound: float = field(repr=False)
    thresh_upper_bound: float = field(repr=False)
    aupimos: torch.Tensor = field(repr=False)  # shape => (N,)

    @property
    def num_images(self) -> int:
        """Number of images."""
        return self.aupimos.shape[0]

    @property
    def num_normal_images(self) -> int:
        """Number of normal images."""
        return int((self.image_classes == 0).sum())

    @property
    def num_anomalous_images(self) -> int:
        """Number of anomalous images."""
        return int((self.image_classes == 1).sum())

    @property
    def image_classes(self) -> torch.Tensor:
        """Image classes (0: normal, 1: anomalous)."""
        # if an instance has `nan` aupimo it's because it's a normal image
        return self.aupimos.isnan().to(torch.int32)

    @property
    def fpr_bounds(self) -> tuple[float, float]:
        """Lower and upper bounds of the FPR integration range."""
        return self.fpr_lower_bound, self.fpr_upper_bound

    @property
    def thresh_bounds(self) -> tuple[float, float]:
        """Lower and upper bounds of the threshold integration range.

        Recall: they correspond to the FPR bounds in reverse order.
        I.e.:
            fpr_lower_bound --> thresh_upper_bound
            fpr_upper_bound --> thresh_lower_bound
        """
        return self.thresh_lower_bound, self.thresh_upper_bound

    def __post_init__(self) -> None:
        """Validate the inputs for the result object are consistent."""
        try:
            _validate.is_rate_range((self.fpr_lower_bound, self.fpr_upper_bound))
            # TODO(jpcbertoldo): warn when it's too low (use parameters from the numpy code)  # noqa: TD003
            _validate.is_num_thresholds_gte2(self.num_thresholds)
            _validate.is_rates(self.aupimos, nan_allowed=True)  # validate is_aupimos

            _validate.validate_threshold_bounds((self.thresh_lower_bound, self.thresh_upper_bound))

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
        """Return an AUPIMO result object from a PIMO result object.

        Args:
            pimo_result: PIMO result object
            fpr_bounds: lower and upper bounds of the FPR integration range
            num_thresholds_auc: number of thresholds used to effectively compute AUPIMO;
                         NOT the number of thresholds used to compute the PIMO curve!
            aupimos: AUPIMO scores
            paths: paths to the source images to which the AUPIMO scores correspond.
        """
        if pimo_result.per_image_tprs.shape[0] != aupimos.shape[0]:
            msg = (
                f"Invalid {cls.__name__} object. Attributes have inconsistent shapes: "
                f"there are {pimo_result.per_image_tprs.shape[0]} PIMO curves but {aupimos.shape[0]} AUPIMO scores."
            )
            raise TypeError(msg)

        if not torch.isnan(aupimos[pimo_result.image_classes == 0]).all():
            msg = "Expected all normal images to have NaN AUPIMOs, but some have non-NaN values."
            raise TypeError(msg)

        if torch.isnan(aupimos[pimo_result.image_classes == 1]).any():
            msg = "Expected all anomalous images to have valid AUPIMOs (not nan), but some have NaN values."
            raise TypeError(msg)

        fpr_lower_bound, fpr_upper_bound = fpr_bounds
        # recall: fpr upper/lower bounds are the same as the thresh lower/upper bounds
        _, thresh_lower_bound, __ = pimo_result.thresh_at(fpr_upper_bound)
        _, thresh_upper_bound, __ = pimo_result.thresh_at(fpr_lower_bound)
        # `_` is the threshold's index, `__` is the actual fpr value
        return cls(
            fpr_lower_bound=fpr_lower_bound,
            fpr_upper_bound=fpr_upper_bound,
            num_thresholds=num_thresholds_auc,
            thresh_lower_bound=float(thresh_lower_bound),
            thresh_upper_bound=float(thresh_upper_bound),
            aupimos=aupimos,
        )
