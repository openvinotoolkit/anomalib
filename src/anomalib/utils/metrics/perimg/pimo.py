"""Per-Image Overlap (PIMO, pronounced pee-mo) curve.

Two variants of AUCs are implemented:
    - AUPImO: Area Under the Per-Image Overlap (PIMO) curves.
              I.e. a metric of per-image average TPR.

for shared fpr = mean( perimg fpr ) == set fpr
    find the th = fpr^-1( MAX_FPR ) with a binary search on the pixels of the norm images
    i.e. it's not necessary to compute the perimg fpr curves (tf. binclf curves) in advance
for other shared fpr alternatives, it's necessary to compute the perimg fpr curves first anyway

further: also choose the th upper bound to be the max score at normal pixels
"""


from __future__ import annotations

from collections import namedtuple

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.pyplot import Figure
from numpy import ndarray
from torch import Tensor

from .binclf_curve import PerImageBinClfCurve
from .common import (
    _perimg_boxplot_stats,
    _validate_atleast_one_anomalous_image,
    _validate_atleast_one_normal_image,
    _validate_nonzero_rate,
)
from .plot import (
    _add_integration_range_to_pimo_curves,
    plot_all_pimo_curves,
    plot_aupimo_boxplot,
    plot_boxplot_pimo_curves,
    plot_pimfpr_curves_norm_only,
    plot_th_fpr_curves_norm_only,
)

# =========================================== METRICS ===========================================

PImOResult = namedtuple(
    "PImOResult",
    [
        "thresholds",
        "fprs",
        "shared_fpr",
        "tprs",
        "image_classes",
    ],
)
PImOResult.__doc__ = """PImO result (from `PImO.compute()`).

[0] thresholds: shape (num_thresholds,), a `float` dtype as given in update()
[1] fprs: shape (num_images, num_thresholds), dtype `float64`, \in [0, 1]
[2] shared_fpr: shape (num_thresholds,), dtype `float64`, \in [0, 1]
[3] tprs: shape (num_images, num_thresholds), dtype `float64`, \in [0, 1] for anom images, `nan` for norm images
[4] image_classes: shape (num_images,), dtype `int32`, \in {0, 1}

- `num_thresholds` is an attribute of `PImO` and is given in the constructor (from parent class).
- `num_images` depends on the data seen by the model at the update() calls.
"""


class PImO(PerImageBinClfCurve):
    """Per-Image Overlap (PIMO, pronounced pee-mo) curve.

    PImO a measure of TP level across multiple thresholds,
        which are indexed by an FP measure on the normal images.

    At a given threshold:
        X-axis: False Positive metric shared across images:
            1. In-image FPR average on normal images (equivalent to the set FPR of normal images).
        Y-axis: Overlap between the class 'anomalous' in the ground truth and the predicted masks (in-image TPR).

    Note about other shared FPR alternatives:
        It can be made harder by using the cross-image max (or high-percentile) FPRs instead of the mean.
        I.e. the shared-fp axis (x-axies) is a statistic (across normal images) at each threshold.
        Rationale: this will further punish models that have exceptional FPs in normal images.
        Rationale: this will further punish models that have exceptional FPs in normal images.

    FP: False Positive
    FPR: False Positive Rate
    TP: True Positive
    TPR: True Positive Rate
    """

    def compute(self) -> PImOResult:  # type: ignore
        """Compute the PImO curve.

        Returns: PImOResult
        See `anomalib.utils.metrics.perimg.pimo.PImOResult` for details.
        """
        if self.is_empty:
            return PImOResult(
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float64),
                torch.empty(0, dtype=torch.float64),
                torch.empty(0, dtype=torch.float64),
                torch.empty(0, dtype=torch.int32),
            )

        thresholds, binclf_curves, image_classes = super().compute()

        _validate_atleast_one_anomalous_image(image_classes)  # necessary for the TPR
        _validate_atleast_one_normal_image(image_classes)  # necessary for the shared FPR

        # (num_images, num_thresholds); from the parent class
        # fprs can be `nan` if an anomalous image is fully covered by the mask
        # but it's ok because we will use only the normal images
        tprs = PerImageBinClfCurve.tprs(binclf_curves)
        fprs = PerImageBinClfCurve.fprs(binclf_curves)

        # see note about shared FPR alternatives in the class's docstring
        shared_fpr = fprs[image_classes == 0].mean(dim=0)  # shape: (num_thresholds,)

        return PImOResult(thresholds, fprs, shared_fpr, tprs, image_classes)

    def plot(
        self,
        ax: Axes | None = None,
    ) -> tuple[Figure | None, Axes]:
        """Plot shared FPR vs Per-Image Overlap (PImO) curves."""

        if self.is_empty:
            return None, None

        _, __, shared_fpr, tprs, image_classes = self.compute()

        fig, ax = plot_all_pimo_curves(
            shared_fpr,
            tprs,
            image_classes,
            ax=ax,
        )
        ax.set_xlabel("Mean FPR on Normal Images")

        return fig, ax


class AUPImO(PImO):
    """Area Under the Per-Image Overlap (PImO) curve.

    AU is computed by the trapezoidal rule, each curve being treated separately.
    """

    def __init__(
        self,
        num_thresholds: int = 10_000,
        ubound: float | Tensor = 1.0,
        **kwargs,
        # TODO remove **kwargs from all metrics (silent bug if not used)
    ) -> None:
        """Area Under the Per-Image Overlap (PImO) curve.

        Args:
            num_thresholds: number of thresholds to use for the binclf curves
                            refer to `anomalib.utils.metrics.perimg.binclf_curve.PerImageBinClfCurve`
            ubound: upper bound of the FPR range to compute the AUC

        """
        super().__init__(num_thresholds=num_thresholds, **kwargs)

        _validate_nonzero_rate(ubound)
        self.register_buffer("ubound", torch.as_tensor(ubound, dtype=torch.float64))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ubound={self.ubound})"

    def compute(self) -> tuple[PImOResult, Tensor]:  # type: ignore
        """Compute the Area Under the Per-Image Overlap curves (AUPImO).

        Returns: (PImOResult, aucs)
            [0] PImOResult: PImOResult, see `anomalib.utils.metrics.perimg.pimo.PImOResult` for details.
            [1] aucs: shape (num_images,), dtype `float64`, \in [0, 1]
        """

        if self.is_empty:
            return PImOResult(
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float64),
                torch.empty(0, dtype=torch.float64),
                torch.empty(0, dtype=torch.float64),
                torch.empty(0, dtype=torch.int32),
            ), torch.empty(0, dtype=torch.float64)

        pimoresult = thresholds, fprs, shared_fpr, tprs, image_classes = super().compute()

        # get the index of the value in `shared_fpr` that is closest to `self.ubound in abs value
        # knwon issue: `shared_fpr[ubound_idx]` might not be exactly `self.ubound`
        # but it's ok because `num_thresholds` should be large enough so that the error is negligible
        ubound_idx = torch.argmin(torch.abs(shared_fpr - self.ubound))

        # limit the curves to the integration range [0, ubound]
        # `shared_fpr` and `tprs` are in descending order; `flip()` reverts to ascending order
        tprs_auc: Tensor = tprs[:, ubound_idx:].flip(dims=(1,))
        shared_fpr_auc: Tensor = shared_fpr[ubound_idx:].flip(dims=(0,))

        aucs: Tensor = torch.trapezoid(tprs_auc, x=shared_fpr_auc, dim=1)

        # normalize the size of `aucs` by dividing by the x-range size
        # clip(0, 1) makes sure that the values are in [0, 1] (in case of numerical errors)
        aucs = (aucs / self.ubound).clip(0, 1)

        return pimoresult, aucs

    def plot_all_pimo_curves(
        self,
        ax: Axes | None = None,
    ) -> tuple[Figure | None, Axes]:
        """Plot shared FPR vs Per-Image Overlap (PImO) curves (all curves).
        Integration range is shown when `self.ubound < 1`.
        """

        if self.is_empty:
            return None, None

        (thresholds, fprs, shared_fpr, tprs, image_classes), aucs = self.compute()

        fig, ax = plot_all_pimo_curves(
            shared_fpr,
            tprs,
            image_classes,
            ax=ax,
        )
        ax.set_xlabel("Mean FPR on Normal Images")

        if self.ubound < 1:
            _add_integration_range_to_pimo_curves(ax, (None, self.ubound))

        return fig, ax

    def boxplot_stats(self) -> list[dict[str, str | int | float | None]]:
        """Compute boxplot stats of AUPImO values (e.g. median, mean, quartiles, etc.).

        Returns:
            list[dict[str, str | int | float | None]]: List of AUCs statistics from a boxplot.
            refer to `anomalib.utils.metrics.perimg.common._perimg_boxplot_stats()` for the keys and values.
        """
        (_, __, ___, ____, image_classes), aucs = self.compute()
        stats = _perimg_boxplot_stats(values=aucs, image_classes=image_classes, only_class=1)
        return stats

    def plot_boxplot_pimo_curves(
        self,
        ax: Axes | None = None,
    ) -> tuple[Figure | None, Axes]:
        """Plot shared FPR vs Per-Image Overlap (PImO) curves (boxplot images only).
        The 'boxplot images' are those from the boxplot of AUPImO values (see `AUPImO.boxplot_stats()`).
        Integration range is shown when `self.ubound < 1`.
        """

        if self.is_empty:
            return None, None

        (thresholds, fprs, shared_fpr, tprs, image_classes), aucs = self.compute()
        fig, ax = plot_boxplot_pimo_curves(
            shared_fpr,
            tprs,
            image_classes,
            self.boxplot_stats(),
            ax=ax,
        )
        ax.set_xlabel("Mean FPR on Normal Images")

        if self.ubound < 1:
            _add_integration_range_to_pimo_curves(ax, (None, self.ubound))

        return fig, ax

    def plot_boxplot(
        self,
        ax: Axes | None = None,
    ) -> tuple[Figure | None, Axes]:
        """Plot boxplot of AUPImO values."""

        if self.is_empty:
            return None, None

        (thresholds, fprs, shared_fpr, tprs, image_classes), aucs = self.compute()
        fig, ax = plot_aupimo_boxplot(aucs, image_classes, ax=ax)
        return fig, ax

    def plot(
        self,
        axes: Axes | ndarray | None = None,
    ) -> tuple[Figure | None, Axes | ndarray]:
        """Plot AUPImO boxplot with its statistics' PImO curves."""

        if self.is_empty:
            return None, None

        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), width_ratios=[6, 8])
            fig.suptitle("Area Under the Per-Image Overlap (AUPImO) Curves")
            fig.set_tight_layout(True)
        else:
            fig, axes = (None, axes)

        if isinstance(axes, Axes):
            return self.plot_boxplot_pimo_curves(ax=axes)

        if not isinstance(axes, ndarray):
            raise ValueError(f"Expected argument `axes` to be a matplotlib Axes or ndarray, but got {type(axes)}.")

        if axes.size != 2:
            raise ValueError(
                f"Expected argument `axes` , when type `ndarray`, to be of size 2, but got size {axes.size}."
            )

        axes = axes.flatten()
        self.plot_boxplot(ax=axes[0])
        axes[0].set_title("AUC Boxplot")
        self.plot_boxplot_pimo_curves(ax=axes[1])
        axes[1].set_title("Curves")
        return fig, axes

    def plot_perimg_fprs(
        self,
        axes: ndarray | None = None,
    ) -> tuple[Figure | None, ndarray]:
        """Plot the AUC boundary conditions based on FPR metrics on normal images.
        TODO review docstring
        """

        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), width_ratios=[6, 8])
            fig.suptitle("AUPImO Integration Boundary Conditions")
            fig.set_tight_layout(True)
        elif not isinstance(axes, ndarray):
            raise ValueError(f"Expected argument `axes` to be an ndarray of matplotlib Axes, but got {type(axes)}.")
        elif axes.size != 2:
            raise ValueError(f"Expected argument `axes` to be of size 2, but got size {axes.size}.")
        else:
            fig, axes = (None, axes)

        axes = axes.flatten()

        (thresholds, fprs, shared_fpr, tprs, image_classes), aucs = self.compute()

        # FRP upper bound is threshold lower bound
        thidx_lbound = torch.argmin(torch.abs(shared_fpr - self.ubound))
        th_lbound = thresholds[thidx_lbound]

        plot_th_fpr_curves_norm_only(fprs, shared_fpr, thresholds, image_classes, th_lbound, self.ubound, ax=axes[0])
        _add_integration_range_to_pimo_curves(axes[0], (None, self.ubound))

        plot_pimfpr_curves_norm_only(fprs, shared_fpr, image_classes, ax=axes[1])

        return fig, axes


class AULogPImO(PImO):
    """Area Under the Per-Image Overlap (PIMO, pronounced pee-mo) curves with log(FPR) (instead of FPR) in the X-axis

    This will further give more importance/visibility to the low FPR region.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("**coming up later**")
        # # TODO log fpr option here
        # # logfpr: whether to use log scale for the FPR axis
        # # logfpr_epsilon: small positive number to avoid `log(0)`; used only if `logfpr` is True
        # _format_axis_rate_metric_log(ax, axis=0, lower_limit=logfpr_epsilon)
