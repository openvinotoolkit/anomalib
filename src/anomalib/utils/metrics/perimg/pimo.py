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

import json
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.pyplot import Figure
from numpy import ndarray
from torch import Tensor

from .binclf_curve import PerImageBinClfCurve
from .common import (
    _validate_and_convert_aucs,
    _validate_and_convert_fpath,
    _validate_and_convert_rate,
    _validate_atleast_one_anomalous_image,
    _validate_atleast_one_normal_image,
    _validate_image_classes,
    _validate_perimg_rate_curves,
    _validate_rate_curve,
    _validate_thresholds,
    perimg_boxplot_stats,
)
from .plot import (
    _add_avline_at_score_random_model,
    _add_integration_range_to_pimo_curves,
    _format_axis_rate_metric_log,
    plot_all_pimo_curves,
    plot_aupimo_boxplot,
    plot_boxplot_logpimo_curves,
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
[1] fprs: shape (num_images, num_thresholds), dtype `float64`, \\in [0, 1]
[2] shared_fpr: shape (num_thresholds,), dtype `float64`, \\in [0, 1]
[3] tprs: shape (num_images, num_thresholds), dtype `float64`, \\in [0, 1] for anom images, `nan` for norm images
[4] image_classes: shape (num_images,), dtype `int32`, \\in {0, 1}

- `num_thresholds` is an attribute of `PImO` and is given in the constructor (from parent class).
- `num_images` depends on the data seen by the model at the update() calls.
"""


def _validate_pimoresult(pimoresult: PImOResult):
    if len(pimoresult) != 5:
        raise ValueError(
            f"Expected argument `pimoresult` to be a namedtuple with 5 elements, but got {len(pimoresult)}."
        )

    thresholds, fprs, shared_fpr, tprs, image_classes = pimoresult
    _validate_thresholds(thresholds)
    _validate_perimg_rate_curves(fprs, nan_allowed=True)  # anomalous images can have nan fprs if fully covered by 1s
    _validate_rate_curve(shared_fpr, nan_allowed=False)
    _validate_perimg_rate_curves(tprs, nan_allowed=True)  # normal images have nan tprs by definition
    _validate_image_classes(image_classes)
    _validate_atleast_one_anomalous_image(image_classes)
    _validate_atleast_one_normal_image(image_classes)
    _validate_perimg_rate_curves(fprs[image_classes == 0], nan_allowed=False)
    _validate_perimg_rate_curves(tprs[image_classes == 1], nan_allowed=False)


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

    @staticmethod
    def _save(pimoresult: PImOResult, fpath: str | Path):
        """Statict method, functional method so children classes can use it."""
        fpath = _validate_and_convert_fpath(fpath, extension=".pt")
        payload_curves = {
            "shared_fpr_metric": "mean_inimage_fpr",
            "thresholds": pimoresult.thresholds,
            "fprs": pimoresult.fprs,
            "shared_fpr": pimoresult.shared_fpr,
            "tprs": pimoresult.tprs,
            "image_classes": pimoresult.image_classes,
        }
        torch.save(payload_curves, fpath)

    def save(self, fpath: str | Path):
        """Save the PImO curve to a `.pt` file.

        Args:
            fpath: path to the file where to save the curve.
        """
        pimoresult = self.compute()
        self._save(pimoresult, fpath)

    @staticmethod
    def load(fpath: str | Path) -> PImOResult:
        """Private method so children classes can use it.
        Args:
            fpath: path to the `.pt` file where to load the PImOResult.
        """
        fpath = _validate_and_convert_fpath(fpath, extension=".pt")
        payload_curves = torch.load(fpath)
        try:
            pimoresult = PImOResult(
                payload_curves["thresholds"],
                payload_curves["fprs"],
                payload_curves["shared_fpr"],
                payload_curves["tprs"],
                payload_curves["image_classes"],
            )
        except KeyError as ex:
            raise KeyError(f"Invalid {PImOResult.__name__}, expected key {ex} in file {fpath}.") from ex
        try:
            _validate_pimoresult(pimoresult)
        except Exception as ex:
            raise RuntimeError(f"Invalid {PImOResult.__name__} in file {fpath}.") from ex

        return pimoresult


class AUPImO(PImO):
    """Area Under the Per-Image Overlap (PImO) curve.

    AU is computed by the trapezoidal rule, each curve being treated separately.
    """

    def __init__(
        self,
        num_thresholds: int = 10_000,
        ubound: float | Tensor = 1.0,
    ) -> None:
        """Area Under the Per-Image Overlap (PImO) curve.

        Args:
            num_thresholds: number of thresholds to use for the binclf curves
                            refer to `anomalib.utils.metrics.perimg.binclf_curve.PerImageBinClfCurve`
            ubound: upper bound of the FPR range to compute the AUC

        """
        super().__init__(num_thresholds=num_thresholds)

        _validate_and_convert_rate(ubound)
        self.register_buffer("ubound", torch.as_tensor(ubound, dtype=torch.float64))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ubound={self.ubound})"

    def compute(self) -> tuple[PImOResult, Tensor]:  # type: ignore
        """Compute the Area Under the Per-Image Overlap curves (AUPImO).

        Returns: (PImOResult, aucs)
            [0] PImOResult: PImOResult, see `anomalib.utils.metrics.perimg.pimo.PImOResult` for details.
            [1] aucs: shape (num_images,), dtype `float64`, \\in [0, 1]
        """

        if self.is_empty:
            return PImOResult(
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float64),
                torch.empty(0, dtype=torch.float64),
                torch.empty(0, dtype=torch.float64),
                torch.empty(0, dtype=torch.int32),
            ), torch.empty(0, dtype=torch.float64)

        pimoresult = _, __, shared_fpr, tprs, ___ = super().compute()

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

        (_, __, shared_fpr, tprs, image_classes), ___ = self.compute()

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
            refer to `anomalib.utils.metrics.perimg.common.perimg_boxplot_stats()` for the keys and values.
        """
        (_, __, ___, ____, image_classes), aucs = self.compute()
        stats = perimg_boxplot_stats(values=aucs, image_classes=image_classes, only_class=1)
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
        _add_avline_at_score_random_model(ax, 0.5)
        return fig, ax

    def plot(
        self,
        ax: Axes | ndarray | None = None,
    ) -> tuple[Figure | None, Axes | ndarray]:
        """Plot AUPImO boxplot with its statistics' PImO curves."""

        if self.is_empty:
            return None, None

        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(14, 6), width_ratios=[6, 8])
            fig.suptitle("Area Under the Per-Image Overlap (AUPImO) Curves")
            fig.set_layout_engine("tight")
        else:
            fig, ax = (None, ax)

        if isinstance(ax, Axes):
            return self.plot_boxplot_pimo_curves(ax=ax)

        if not isinstance(ax, ndarray):
            raise ValueError(f"Expected argument `ax` to be a matplotlib Axes or ndarray, but got {type(ax)}.")

        if ax.size != 2:
            raise ValueError(f"Expected argument `ax` , when type `ndarray`, to be of size 2, but got size {ax.size}.")

        ax = ax.flatten()
        self.plot_boxplot(ax=ax[0])
        ax[0].set_title("AUC Boxplot")
        self.plot_boxplot_pimo_curves(ax=ax[1])
        ax[1].set_title("Curves")
        return fig, ax

    def plot_perimg_fprs(
        self,
        ax: ndarray | None = None,
    ) -> tuple[Figure | None, ndarray]:
        """Plot the AUC boundary conditions based on FPR metrics on normal images.

        Args:
            ax: ndarray of matplotlib Axes of size 2, or None.
                If None, the function will create the ax.
        Returns:
            tuple[Figure | None, ndarray]: (fig, ax)
                fig: matplotlib Figure
                ax: ndarray of matplotlib Axes of size 2
        """

        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(14, 6), width_ratios=[6, 8])
            fig.suptitle("AUPImO Integration Boundary Conditions")
            fig.set_tight_layout(True)
        elif not isinstance(ax, ndarray):
            raise ValueError(f"Expected argument `ax` to be an ndarray of matplotlib Axes, but got {type(ax)}.")
        elif ax.size != 2:
            raise ValueError(f"Expected argument `ax` to be of size 2, but got size {ax.size}.")
        else:
            fig, ax = (None, ax)

        ax = ax.flatten()

        (thresholds, fprs, shared_fpr, tprs, image_classes), aucs = self.compute()

        # FRP upper bound is threshold lower bound
        thidx_lbound = torch.argmin(torch.abs(shared_fpr - self.ubound))
        th_lbound = thresholds[thidx_lbound]

        plot_th_fpr_curves_norm_only(
            fprs, shared_fpr, thresholds, image_classes, th_lb_fpr_ub=(th_lbound, self.ubound), ax=ax[0]
        )

        plot_pimfpr_curves_norm_only(fprs, shared_fpr, image_classes, ax=ax[1])
        _add_integration_range_to_pimo_curves(ax[1], (None, self.ubound))

        return fig, ax

    @staticmethod
    def _deduce_curves_fpath(fpath: Path):
        return fpath.parent / f"{fpath.stem}_curves.pt"

    def save(self, fpath: str | Path, curve: bool = True):
        """Save the AUPImO values (AUCs) in a JSON file.
        Optionally, save the AUPImO curves in a `.pt` file.

        Args:
            fpath: path to the file where to save the AUPImO values.
            curve: whether to save the AUPImO curves in a `.pt` file.
        """
        fpath = _validate_and_convert_fpath(fpath, extension=".json")
        pimoresult, aucs = self.compute()
        payload = {
            "ubound": self.ubound.item(),
            "aupimo": aucs.tolist(),
        }
        with fpath.open("w") as f:
            json.dump(payload, f, indent=4)
        if curve:
            fpath_curves = AUPImO._deduce_curves_fpath(fpath)
            PImO._save(pimoresult, fpath_curves)
            return fpath_curves

    @staticmethod
    def load(fpath: str | Path, curve: bool = True) -> Tensor | tuple[PImOResult, Tensor]:  # type: ignore
        """Load the AUPImO values (AUCs) from a JSON file.
        Optionally, load the PImO curves from a `.pt` file (file name is deduced from the JSON file name).

        Args:
            fpath: path to the file where to load the AUPImO values.
            curve: whether to load the PImO curves from a `.pt` file.
        Returns:
            Tensor: AUPImO values (AUCs)
            or
            tuple[PImOResult, Tensor]: (PImOResult, AUPImO values (AUCs)).
        """

        fpath = _validate_and_convert_fpath(fpath, extension=".json")
        with fpath.open("r") as f:
            loaded = json.load(f)
        try:
            ubound = loaded["ubound"]
            aupimo = loaded["aupimo"]
        except KeyError as ex:
            raise KeyError(f"Invalid {AUPImO.__name__} saved AUCs, expected key {ex} in file {fpath}.") from ex
        try:
            loaded["ubound"] = _validate_and_convert_rate(ubound, nonzero=True)
            loaded["aupimo"] = _validate_and_convert_aucs(aupimo, nan_allowed=True).to(torch.float64)
        except Exception as ex:
            raise RuntimeError(f"Invalid {AUPImO.__name__} saved AUCs in file {fpath}.") from ex
        if curve:
            curves_fpath = AUPImO._deduce_curves_fpath(fpath)
            pimoresult = PImO.load(curves_fpath)
            return pimoresult, loaded
        return loaded


class AULogPImO(PImO):
    """Area Under the Log Per-Image Overlap (LogPIMO, pronounced log pee-mo).

    LogPImO curves have log(FPR) in the X-axis (instead of FPR).

    AULogPImO's primitive (to be normalized) is

        \\integral_{L}^{U} TPR(FPR) dlog(FPR) = \\integral_{log(L)}^{log(U)} TPR(FPR) FPR^{-1} dFPR

    L: FPR lower bound \\in (0, 1)
    U: FPR upper bound \\in (0, 1] such that U > L
    FPR: False Positive Rate
    TPR: True Positive Rate

    F \\in [L, U]^N is a sequence of `N` FPRs, and T \\in [0, 1]^N is a vector of `N` TPRs,
    such that F_{i+1} > F_i for all i \\in [1, N-1], and T_i = TPR(F_i) for i = 1, ..., N.

    LogF \\in (-inf, 1]^N is a sequence of `N` log(FPR)s; i.e. LogF_i = log(F_i) for i = 1, ..., N.

    The integral is computed by the trapezoidal rule, each curve being treated separately.

    It can be computed in two ways:
        (1) trapezoid(F, T / F), where / is element-wise division
        (2) trapezoid(LogF, T)

    We use (2) and normalize the value to have a score that is in [0, 1].
    The normalization constant is the score of the perfect model (TPR = 1 for all FPRs):

    MAXAUC = \\integral_{U}^{L} FRP^{-1} dFPR = log(U) - log(L) = log(U/L)
    MAXAUC = log(U/L)

    AULogPImO = trapezoid(LogF, T) / log(U/L)

    TODO make functional of random model score
    """

    def __init__(
        self,
        num_thresholds: int = 10_000,
        lbound: float | Tensor = 1e-3,
        ubound: float | Tensor = 1.0,
    ) -> None:
        """Area Under the Per-Image Overlap (PImO) curve.

        Args:
            num_thresholds: number of thresholds to use for the binclf curves
                            refer to `anomalib.utils.metrics.perimg.binclf_curve.PerImageBinClfCurve`
            lbound: lower bound of the FPR range to compute the AUC
            ubound: upper bound of the FPR range to compute the AUC

                AUC is computed by the trapezoidal rule, each curve being treated separately,
                in the range [lbound, ubound].

        """
        super().__init__(num_thresholds=num_thresholds)

        _validate_and_convert_rate(lbound)
        _validate_and_convert_rate(ubound)

        if lbound >= ubound:
            raise ValueError(f"Expected argument `lbound` to be < `ubound`, but got {lbound} >= {ubound}.")

        self.register_buffer("lbound", torch.as_tensor(lbound, dtype=torch.float64))
        self.register_buffer("ubound", torch.as_tensor(ubound, dtype=torch.float64))
        # TODO add a warning for FPR lower bound too low compared to the inputs' resolution
        # if that's the case, the FPR levels will jump too much, the error when finding the numerical
        # `fpr[argmin(abs(fpr - FPR))]` will be important

    @property
    def max_primitive_auc(self) -> float:
        """Maximum AUC value of the primitive integral (before normalization)."""
        return torch.log(self.ubound / self.lbound).item()

    @property
    def random_model_primitive_auc(self) -> float:
        """AUC value of the primitive integral (before normalization) of a random model.
        Random model: TPR = FPR for all FPRs.
        """
        return self.ubound.item() - self.lbound.item()

    @property
    def random_model_auc(self) -> float:
        """AUC value of a random model.
        Random model: TPR = FPR for all FPRs.
        """
        return self.random_model_primitive_auc / self.max_primitive_auc

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lbound={self.lbound}, ubound={self.ubound})"

    def compute(self) -> tuple[PImOResult, Tensor]:  # type: ignore
        """Compute the Area Under the Log Per-Image Overlap curves (AULogPImO).

        Returns: (PImOResult, aucs)
            [0] PImOResult: PImOResult, see `anomalib.utils.metrics.perimg.pimo.PImOResult` for details.
            [1] aucs: shape (num_images,), dtype `float64`, \\in [0, 1]
        """

        if self.is_empty:
            return PImOResult(
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float64),
                torch.empty(0, dtype=torch.float64),
                torch.empty(0, dtype=torch.float64),
                torch.empty(0, dtype=torch.int32),
            ), torch.empty(0, dtype=torch.float64)

        pimoresult = thresholds, _, shared_fpr, tprs, __ = super().compute()

        # get the index of the value in `shared_fpr` that is closest to `self.ubound in abs value
        # knwon issue: `shared_fpr[ubound_idx]` might not be exactly `self.ubound`
        # but it's ok because `num_thresholds` should be large enough so that the error is negligible
        ubound_th_idx = torch.argmin(torch.abs(shared_fpr - self.ubound))
        lbound_th_idx = torch.argmin(torch.abs(shared_fpr - self.lbound))

        # deal with edge cases
        # reminder: fpr lower/upper bound is threshold upper/lower bound
        if ubound_th_idx >= lbound_th_idx:
            raise RuntimeError(
                "Expected `lbound` and `ubound` to be such that `threshold(ubound) < threshold(lbound)`, "
                f"but got `threshold(ubound) = {thresholds[ubound_th_idx]}` >= "
                f"`threshold(lbound) = {thresholds[lbound_th_idx]}`."
            )

        # limit the curves to the integration range [lbound, ubound]
        # `shared_fpr` and `tprs` are in descending order; `flip()` reverts to ascending order
        tprs_auc: Tensor = tprs[:, ubound_th_idx : (lbound_th_idx + 1)].flip(dims=(1,))
        shared_fpr_auc: Tensor = shared_fpr[ubound_th_idx : (lbound_th_idx + 1)].flip(dims=(0,))
        # as described in the class's docstring:
        shared_fpr_auc = shared_fpr_auc.log()

        aucs: Tensor = torch.trapezoid(tprs_auc, x=shared_fpr_auc, dim=1)

        # normalize, then clip(0, 1) makes sure that the values are in [0, 1] in case of numerical errors
        aucs = (aucs / self.max_primitive_auc).clip(0, 1)

        return pimoresult, aucs

    def plot_all_logpimo_curves(
        self,
        ax: Axes | None = None,
    ) -> tuple[Figure | None, Axes]:
        """Plot log10( shared FPR ) vs Per-Image Overlap (LogPImO) curves (all curves)."""

        if self.is_empty:
            return None, None

        (_, __, shared_fpr, tprs, image_classes), ___ = self.compute()

        fig, ax = plot_all_pimo_curves(
            shared_fpr,
            tprs,
            image_classes,
            ax=ax,
        )
        ax.set_xlabel("Log10 of Mean FPR on Normal Images")
        ax.set_title("Log Per-Image Overlap (LogPImO) Curves")
        _format_axis_rate_metric_log(ax, axis=0, lower_lim=self.lbound, upper_lim=self.ubound)
        # they are not exactly the same as the input because the function above rounds them
        xtickmin, xtickmax = ax.xaxis.get_ticklocs()[[0, -1]]
        _add_integration_range_to_pimo_curves(
            ax, (self.lbound, self.ubound), span=(xtickmin < self.lbound or xtickmax > self.ubound)
        )

        return fig, ax

    def boxplot_stats(self) -> list[dict[str, str | int | float | None]]:
        """Compute boxplot stats of AULogPImO values (e.g. median, mean, quartiles, etc.).

        Returns:
            list[dict[str, str | int | float | None]]: List of AUCs statistics from a boxplot.
            refer to `anomalib.utils.metrics.perimg.common.perimg_boxplot_stats()` for the keys and values.
        """
        (_, __, ___, ____, image_classes), aucs = self.compute()
        stats = perimg_boxplot_stats(values=aucs, image_classes=image_classes, only_class=1)
        return stats

    def plot_boxplot_logpimo_curves(
        self,
        ax: Axes | None = None,
    ) -> tuple[Figure | None, Axes]:
        """Plot log10( shared FPR ) vs Per-Image Overlap (LogPImO) curves (boxplot images only).
        The 'boxplot images' are those from the boxplot of AULogPImO values (see `AULogPImO.boxplot_stats()`).
        """

        if self.is_empty:
            return None, None

        (_, __, shared_fpr, tprs, image_classes), ___ = self.compute()
        fig, ax = plot_boxplot_logpimo_curves(
            shared_fpr,
            tprs,
            image_classes,
            self.boxplot_stats(),
            self.lbound,
            self.ubound,
            ax=ax,
        )
        ax.set_xlabel("Log10 of Mean FPR on Normal Images")
        return fig, ax

    def plot_boxplot(
        self,
        ax: Axes | None = None,
    ) -> tuple[Figure | None, Axes]:
        """Plot boxplot of AULogPImO values."""

        if self.is_empty:
            return None, None

        (_, __, ___, ____, image_classes), aucs = self.compute()
        fig, ax = plot_aupimo_boxplot(aucs, image_classes, ax=ax)
        ax.set_xlabel("AULogPImO [%]")
        ax.set_title("Area Under the Log Per-Image Overlap (AULogPImO) Boxplot")
        _add_avline_at_score_random_model(ax, self.random_model_auc)
        return fig, ax

    def plot(
        self,
        ax: Axes | ndarray | None = None,
    ) -> tuple[Figure | None, Axes | ndarray]:
        """Plot AULogPImO boxplot with its statistics' LogPImO curves."""

        if self.is_empty:
            return None, None

        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(14, 6), width_ratios=[6, 8])
            fig.suptitle("Area Under the Log Per-Image Overlap (AULogPImO) Curves")
            fig.set_tight_layout(True)
        else:
            fig, ax = (None, ax)

        if isinstance(ax, Axes):
            return self.plot_boxplot_logpimo_curves(ax=ax)

        if not isinstance(ax, ndarray):
            raise ValueError(f"Expected argument `ax` to be a matplotlib Axes or ndarray, but got {type(ax)}.")

        if ax.size != 2:
            raise ValueError(f"Expected argument `ax` , when type `ndarray`, to be of size 2, but got size {ax.size}.")

        ax = ax.flatten()
        self.plot_boxplot(ax=ax[0])
        self.plot_boxplot_logpimo_curves(ax=ax[1])

        if fig is not None:  # it means the ax were created by this function (and so was the suptitle)
            ax[0].set_title("AUC Boxplot")
            ax[1].set_title("Curves")

        return fig, ax

    def plot_perimg_fprs(
        self,
        ax: ndarray | None = None,
    ) -> tuple[Figure | None, ndarray]:
        """Plot the AUC boundary conditions based on log(FPR) on normal images.
        Args:
            ax: ndarray of matplotlib Axes of size 2, or None.
                If None, the function will create the ax.
        Returns:
            tuple[Figure | None, ndarray]: (fig, ax)
                fig: matplotlib Figure
                ax: ndarray of matplotlib Axes of size 2
        """

        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(14, 6), width_ratios=[6, 8])
            fig.suptitle("AULogPImO Integration Boundary Conditions")
            fig.set_tight_layout(True)
        elif not isinstance(ax, ndarray):
            raise ValueError(f"Expected argument `ax` to be an ndarray of matplotlib Axes, but got {type(ax)}.")
        elif ax.size != 2:
            raise ValueError(f"Expected argument `ax` to be of size 2, but got size {ax.size}.")
        else:
            fig, ax = (None, ax)

        ax = ax.flatten()

        (thresholds, fprs, shared_fpr, _, image_classes), __ = self.compute()

        # FRP upper bound is threshold lower bound
        thidx_lbound = torch.argmin(torch.abs(shared_fpr - self.ubound))
        th_lbound = thresholds[thidx_lbound]

        # FPR lower bound is threshold upper bound
        thidx_ubound = torch.argmin(torch.abs(shared_fpr - self.lbound))
        th_ubound = thresholds[thidx_ubound]

        plot_th_fpr_curves_norm_only(
            fprs,
            shared_fpr,
            thresholds,
            image_classes,
            th_lb_fpr_ub=(th_lbound, self.ubound),
            th_ub_fpr_lb=(th_ubound, self.lbound),
            ax=ax[0],
        )

        plot_pimfpr_curves_norm_only(fprs, shared_fpr, image_classes, ax=ax[1])
        _add_integration_range_to_pimo_curves(ax[1], (self.lbound, self.ubound))

        return fig, ax

    @staticmethod
    def _deduce_curves_fpath(fpath: Path):
        return fpath.parent / f"{fpath.stem}_curves.pt"

    def save(self, fpath: str | Path, curve: bool = True):
        """Save the AULogPImO values (AUCs) in a JSON file.
        Optionally, save the AUPImO curves in a `.pt` file.

        Args:
            fpath: path to the file where to save the AULogPImO values.
            curve: whether to save the AUPImO curves in a `.pt` file.
        """
        fpath = _validate_and_convert_fpath(fpath, extension=".json")
        pimoresult, aucs = self.compute()
        payload = {
            "lbound": self.lbound.item(),
            "ubound": self.ubound.item(),
            "aulogpimo": aucs.tolist(),
        }
        with fpath.open("w") as f:
            json.dump(payload, f, indent=4)
        if curve:
            fpath_curves = AULogPImO._deduce_curves_fpath(fpath)
            PImO._save(pimoresult, fpath_curves)
            return fpath_curves

    @staticmethod
    def load(fpath: str | Path, curve: bool = True) -> Tensor | tuple[PImOResult, Tensor]:  # type: ignore
        """Load the AULogPImO values (AUCs) from a JSON file.
        Optionally, load the PImO curves from a `.pt` file (file name is deduced from the JSON file name).

        Args:
            fpath: path to the file where to load the AULogPImO values.
            curve: whether to load the PImO curves from a `.pt` file.
        Returns:
            Tensor: AULogPImO values (AUCs)
            or
            tuple[PImOResult, Tensor]: (PImOResult, AULogPImO values (AUCs)).
        """

        fpath = _validate_and_convert_fpath(fpath, extension=".json")
        with fpath.open("r") as f:
            loaded = json.load(f)
        try:
            lbound = loaded["lbound"]
            ubound = loaded["ubound"]
            aulogpimo = loaded["aulogpimo"]
        except KeyError as ex:
            raise KeyError(f"Invalid {AULogPImO.__name__} saved AUCs, expected key {ex} in file {fpath}.") from ex
        try:
            loaded["lbound"] = lbound = _validate_and_convert_rate(lbound, nonzero=True, nonone=True)
            loaded["ubound"] = ubound = _validate_and_convert_rate(ubound, nonzero=True)
            if lbound >= ubound:
                raise ValueError(f"Expected argument `lbound` to be < `ubound`, but got {lbound} >= {ubound}.")
            loaded["aulogpimo"] = _validate_and_convert_aucs(aulogpimo, nan_allowed=True).to(torch.float64)
        except Exception as ex:
            raise RuntimeError(f"Invalid {AULogPImO.__name__} saved AUCs in file {fpath}.") from ex
        if curve:
            curves_fpath = AULogPImO._deduce_curves_fpath(fpath)
            pimoresult = PImO.load(curves_fpath)
            return pimoresult, loaded
        return loaded
