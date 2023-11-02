"""Per-Image Overlap curve (PImO, pronounced pee-mo).

PImO is a measure of average True Positive (TP) level across multiple anomaly score thresholds.
The anomaly score thresholds are indexed by a False Positive (FP) measure on the normal images.

Each *anomalous* image has its own curve such that the X-axis is shared by all of them.

At a given threshold:
    X-axis: False Positive metric shared across images.
        1. Average of per-image FP Rate (FPR) on normal images.
        2. Log10 of the above -- curve referred to as `LogPImO`.
    Y-axis: per-image TP Rate (TPR), or "Overlap" between the ground truth and the predicted masks.

Two variants of AUCs are implemented:
    - AUPImO: Area Under the PImO curves.
    - AULogPImO: Area Under the LogPIMO curves.

`AUPImO` can be (optinally) bounded by a maximum shared FPR value (e.g. 30%, default is 100%),
    and the final score is normalized to [0, 1]. The score of a random model is 50%.

`AULogPImO` *can* be (optinally) bounded by a maximum shared FPR value (e.g. 30%, default is 100%),
    it *has* to be bounded by a minimum shared FPR value (e.g. 0.1%), and the final score is normalized to [0, 1].
    The score of a random model depends on the bounds and a helper function in the class is provided to compute it.
"""


from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.pyplot import Figure
from numpy import ndarray
from torch import Tensor

from anomalib.data.utils.image import duplicate_filename

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
    plot_aulogpimo_boxplot,
    plot_aupimo_boxplot,
    plot_boxplot_logpimo_curves,
    plot_boxplot_pimo_curves,
    plot_pimfpr_curves_norm_only,
    plot_th_fpr_curves_norm_only,
)

# =========================================== CONSTANTS ===========================================


class SharedFPRMetric:
    """Shared FPR metric (x-axis of the PImO curve).
    A collection of constants to be used as the `shared_fpr_metric` argument of `PImO`.
    """

    MEAN_PERIMAGE_FPR: ClassVar[str] = "mean_perimage_fpr"


class SharedFPRScale:
    """Shared FPR scale (x-axis of the PImO curve).
    A collection of constants to be used as the `shared_fpr_scale` argument of `PImO`.
    """

    LINEAR: ClassVar[str] = "linear"
    LOG: ClassVar[str] = "log"


# =========================================== METRICS ===========================================


class InvalidPImOResult(ValueError):
    """Something is inconsistent in the PImO result."""

    pass


# TODO review where this is used (check where compute() is called) and use it as type hint
@dataclass
class PImOResult:
    """PImO result (from `PImO.compute()`).

    The attribute `shared_fpr_metric` is a user-defined parameter of the curve.

    thresholds: shape (num_thresholds,), a `float` dtype as given in update()
    fprs: shape (num_images, num_thresholds), dtype `float64`, \\in [0, 1]
    shared_fpr: shape (num_thresholds,), dtype `float64`, \\in [0, 1]
    tprs: shape (num_images, num_thresholds), dtype `float64`, \\in [0, 1] for anom images, `nan` for norm images
    image_classes: shape (num_images,), dtype `int32`, \\in {0, 1}

    - `num_thresholds` comes from `PImO` and is given in the constructor (from parent class).
    - `num_images` depends on the data seen by the model at the update() calls.
    """

    # params (user input)
    shared_fpr_metric: str

    # results (computed)
    thresholds: Tensor = field(repr=False)
    fprs: Tensor = field(repr=False)
    shared_fpr: Tensor = field(repr=False)
    tprs: Tensor = field(repr=False)
    image_classes: Tensor = field(repr=False)

    @property
    def num_thresholds(self) -> int:
        return self.thresholds.shape[0]

    @property
    def num_images(self) -> int:
        return self.image_classes.shape[0]

    def __post_init__(self):
        if len(self.shared_fpr_metric) == 0:
            raise InvalidPImOResult(f"Invalid {self.__class__.__name__} object. `shared_fpr_metric` cannot be empty.")

        try:
            _validate_thresholds(self.thresholds)
            # anomalous images can have nan fprs if fully covered by 1s
            _validate_perimg_rate_curves(self.fprs, nan_allowed=True)
            _validate_rate_curve(self.shared_fpr, nan_allowed=False)
            # normal images have nan tprs by definition
            _validate_perimg_rate_curves(self.tprs, nan_allowed=True)
            _validate_image_classes(self.image_classes)
            _validate_atleast_one_anomalous_image(self.image_classes)
            _validate_atleast_one_normal_image(self.image_classes)
            _validate_perimg_rate_curves(self.fprs[self.image_classes == 0], nan_allowed=False)
            _validate_perimg_rate_curves(self.tprs[self.image_classes == 1], nan_allowed=False)

        except Exception as ex:
            raise InvalidPImOResult(f"Invalid {self.__class__.__name__} object. {ex}") from ex

        # `self.thresholds` and `self.image_classes` have been validated so
        # they are used to validate the other attributes' shapes (`self.num_thresholds` and `self.num_images`)

        if self.fprs.shape != (self.num_images, self.num_thresholds):
            raise InvalidPImOResult(
                f"Invalid {self.__class__.__name__} object. Attributes have inconsistent shapes: "
                f"thresholds.shape={self.thresholds.shape}, image_classes.shape={self.image_classes.shape}, "
                f"but fprs.shape={self.fprs.shape}."
            )

        if self.tprs.shape != (self.num_images, self.num_thresholds):
            raise InvalidPImOResult(
                f"Invalid {self.__class__.__name__} object. Attributes have inconsistent shapes: "
                f"thresholds.shape={self.thresholds.shape}, image_classes.shape={self.image_classes.shape}, "
                f"but tprs.shape={self.tprs.shape}."
            )

        if self.shared_fpr.shape != (self.num_thresholds,):
            raise InvalidPImOResult(
                f"Invalid {self.__class__.__name__} object. Attributes have inconsistent shapes: "
                f"thresholds.shape={self.thresholds.shape}, but shared_fpr.shape={self.shared_fpr.shape}."
            )

    def threshold_index_at(self, shared_fpr_value: float | Tensor) -> int:
        """Return the index of the threshold at the given shared FPR value.

        Args:
            shared_fpr_value: shared FPR value at which to get the threshold index.

        Returns:
            int: index of the threshold at the given shared FPR value.
        """

        shared_fpr_value = _validate_and_convert_rate(shared_fpr_value, nonone=False, nonzero=False)

        if shared_fpr_value < self.shared_fpr.min():
            raise ValueError(
                f"Invalid `shared_fpr_value`. Expected a value in [{self.shared_fpr.min()}, {self.shared_fpr.max()}] "
                f"but got {shared_fpr_value}."
            )

        if shared_fpr_value > self.shared_fpr.max():
            raise ValueError(
                f"Invalid `shared_fpr_value`. Expected a value in [{self.shared_fpr.min()}, {self.shared_fpr.max()}] "
                f"but got {shared_fpr_value}."
            )

        return int(torch.argmin(torch.abs(self.shared_fpr - shared_fpr_value)).item())

    def threshold_at(self, shared_fpr_value: float | Tensor) -> Tensor:
        """Return the threshold at the given shared FPR value.

        Args:
            shared_fpr_value: shared FPR value at which to get the threshold.

        Returns:
            Tensor: 0D tensor, threshold at the given shared FPR value.
        """
        # validations are done in `self.threshold_index_at()`
        idx = self.threshold_index_at(shared_fpr_value)
        return self.thresholds[idx]

    def to_tuple(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return computed attributes as a tuple. Useful to unpack them as variables."""
        return self.thresholds, self.fprs, self.shared_fpr, self.tprs, self.image_classes

    def to_dict(self) -> dict[str, Tensor | str]:
        return {
            # params (user input)
            "shared_fpr_metric": self.shared_fpr_metric,
            # results (computed)
            "thresholds": self.thresholds,
            "fprs": self.fprs,
            "shared_fpr": self.shared_fpr,
            "tprs": self.tprs,
            "image_classes": self.image_classes,
        }

    @classmethod
    def from_dict(cls, dic: dict[str, Tensor | str]) -> "PImOResult":
        try:
            obj = cls(
                dic["shared_fpr_metric"],
                dic["thresholds"],
                dic["fprs"],
                dic["shared_fpr"],
                dic["tprs"],
                dic["image_classes"],
            )

        except KeyError as ex:
            raise InvalidPImOResult(f"Invalid {cls.__name__}, expected key not in dictionary.") from ex

        except InvalidPImOResult as ex:
            raise InvalidPImOResult(f"Invalid {cls.__name__} object from dictionary. {ex}") from ex

        return obj

    def save(self, fpath: str | Path) -> None:
        """Save the PImO result to a `.pt` file.

        Args:
            fpath: path to the `.pt` file where to save the PImO result.
                - must have a `.pt` extension or no extension (in which case `.pt` is added)
                - if the file already exists, a numerical suffix is added to the filename
        """
        fpath = _validate_and_convert_fpath(fpath, extension=".pt")
        fpath = duplicate_filename(fpath)
        payload = self.to_dict()
        torch.save(payload, fpath)

    @classmethod
    def load(cls, fpath: str | Path) -> "PImOResult":
        """Load the PImO result from a `.pt` file.

        Args:
            fpath: path to the `.pt` file where to load the PImOResult.
        """
        fpath = _validate_and_convert_fpath(fpath, extension=".pt")
        payload = torch.load(fpath)
        if not isinstance(payload, dict):
            raise InvalidPImOResult(f"Invalid {cls.__name__} object from file {fpath}, expected a dictionary.")
        try:
            return cls.from_dict(payload)
        except InvalidPImOResult as ex:
            raise InvalidPImOResult(f"Could not load {cls.__name__} from file {fpath}") from ex


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

    def compute(self) -> PImOResult:  # type: ignore[override]
        """Compute the PImO curve.

        Returns: PImOResult
        See `anomalib.utils.metrics.perimg.pimo.PImOResult` for details.
        """

        if self.is_empty:
            raise RuntimeError("Cannot compute PImO curve without any data.")

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

        return PImOResult(
            # `MEAN_PERIMAGE_FPR` is the only one implemented for now
            SharedFPRMetric.MEAN_PERIMAGE_FPR,
            thresholds,
            fprs,
            shared_fpr,
            tprs,
            image_classes,
        )

    def plot(
        self,
        ax: Axes | None = None,
    ) -> tuple[Figure | None, Axes]:
        """Plot shared FPR vs Per-Image Overlap (PImO) curves."""

        if self.is_empty:
            return None, None

        _, __, shared_fpr, tprs, image_classes = self.compute().to_tuple()

        fig, ax = plot_all_pimo_curves(
            shared_fpr,
            tprs,
            image_classes,
            ax=ax,
        )
        # `MEAN_PERIMAGE_FPR` is the only one implemented for now
        ax.set_xlabel("Mean FPR on Normal Images")

        return fig, ax

    # TODO find where this is used and replace it
    def save(self, fpath: str | Path):
        raise NotImplementedError("This method was deleted. Use `PImOResult.save()` instead.")

    # TODO find where this is used and replace it
    @staticmethod
    def load(fpath: str | Path) -> PImOResult:
        raise NotImplementedError("This method was deleted. Use `PImOResult.from_path()` instead.")


class InvalidAUPImOResult(ValueError):
    """Something is inconsistent in the AUPImO result."""

    pass


@dataclass
class AUPImOResult:
    """Area Under the Per-Image Overlap (PImO) curve.

    The attributes `shared_fpr_metric`, `shared_fpr_scale`, `lbound`, and `ubound` are
    user-defined parameters of the metric.

    aucs: shape (num_images,), dtype `float64`, \\in [0, 1] for anom images, `nan` for norm images
    """

    # params (user input)
    shared_fpr_metric: str
    shared_fpr_scale: str
    lbound: Tensor
    ubound: Tensor

    # results (computed)
    lbound_threshold: Tensor
    ubound_threshold: Tensor
    aucs: Tensor = field(repr=False)

    @property
    def image_classes(self) -> Tensor:
        """Image classes are deduced from the AUCs (normals are `nan`)."""
        return self.aucs.isnan().logical_not().to(torch.int32)

    @property
    def num_images(self) -> int:
        return self.aucs.shape[0]

    def __post_init__(self):
        if len(self.shared_fpr_metric) == 0:
            raise InvalidAUPImOResult(f"Invalid {self.__class__.__name__} object. `shared_fpr_metric` cannot be empty.")

        if len(self.shared_fpr_scale) == 0:
            raise InvalidAUPImOResult(f"Invalid {self.__class__.__name__} object. `shared_fpr_scale` cannot be empty.")

        try:
            if self.shared_fpr_scale == SharedFPRScale.LOG:
                self.lbound = _validate_and_convert_rate(self.lbound, nonone=True, nonzero=True).to(torch.float64)
            else:
                self.lbound = _validate_and_convert_rate(self.lbound, nonone=True, nonzero=False).to(torch.float64)
            self.ubound = _validate_and_convert_rate(self.ubound, nonzero=True, nonone=False).to(torch.float64)

            self.lbound_threshold = torch.as_tensor(self.lbound_threshold, dtype=torch.float32)
            self.ubound_threshold = torch.as_tensor(self.ubound_threshold, dtype=torch.float32)

            self.aucs = _validate_and_convert_aucs(self.aucs, nan_allowed=True).to(torch.float64)

            # TODO _validate_and_convert_threshold()

        except ValueError as ex:
            raise InvalidAUPImOResult(f"Invalid {self.__class__.__name__} object.") from ex

        if self.lbound >= self.ubound:
            raise InvalidAUPImOResult(
                f"Invalid {self.__class__.__name__} object. Lower bound must be strictly smaller than upper bound."
            )

        # reminder: fpr lower/upper bound is threshold upper/lower bound
        if self.ubound_threshold >= self.lbound_threshold:
            raise InvalidAUPImOResult(
                f"Invalid {self.__class__.__name__} object. "
                "Upper bound threshold must be strictly smaller than lower bound threshold."
            )

    def validate_consistency_with_curves(self, curves: PImOResult) -> None:
        if self.shared_fpr_metric != curves.shared_fpr_metric:
            raise InvalidAUPImOResult(
                f"Inconsistent {self.__class__.__name__} object with respective {curves.__class__.__name__} object. "
                "Expected `shared_fpr_metric` to be the same but got, respectively, "
                f"{self.shared_fpr_metric} and {curves.shared_fpr_metric}."
            )

        if self.num_images != curves.num_images:
            raise InvalidAUPImOResult(
                f"Inconsistent {self.__class__.__name__} object with respective {curves.__class__.__name__} object. "
                "Expected `num_images` to be the same but got, respectively, "
                f"{self.num_images} and {curves.num_images}."
            )

        if (self.image_classes != curves.image_classes).any():
            raise InvalidAUPImOResult(
                f"Inconsistent {self.__class__.__name__} object with respective {curves.__class__.__name__} object. "
                "Expected `image_classes` to be the same but got, different values."
            )

    def to_dict(self, keeptensor: bool = False) -> dict[str, float | str]:
        lbound = self.lbound if keeptensor else self.lbound.item()
        ubound = self.ubound if keeptensor else self.ubound.item()
        lbound_threshold = self.lbound_threshold if keeptensor else self.lbound_threshold.item()
        ubound_threshold = self.ubound_threshold if keeptensor else self.ubound_threshold.item()
        aucs = self.aucs if keeptensor else self.aucs.tolist()

        return {
            # params (user input)
            "shared_fpr_metric": self.shared_fpr_metric,
            "shared_fpr_scale": self.shared_fpr_scale,
            "lbound": lbound,
            "ubound": ubound,
            # results (computed)
            "lbound_threshold": lbound_threshold,
            "ubound_threshold": ubound_threshold,
            "aucs": aucs,
        }

    @classmethod
    def from_dict(cls, dic: dict[str, float | str] | dict[str, Tensor | str]) -> "AUPImOResult":
        try:
            obj = cls(
                # params (user input)
                dic["shared_fpr_metric"],  # type: ignore
                dic["shared_fpr_scale"],  # type: ignore
                dic["lbound"],
                dic["ubound"],
                # results (computed)
                dic["lbound_threshold"],
                dic["ubound_threshold"],
                dic["aucs"],
            )

        except KeyError as ex:
            raise InvalidAUPImOResult(f"Invalid {cls.__name__}, expected key not in dictionary.") from ex

        except InvalidAUPImOResult as ex:
            raise InvalidAUPImOResult(f"Invalid {cls.__name__} object from dictionary. {ex}") from ex

        return obj

    def save(self, fpath: str | Path) -> None:
        """Save the AUPImO result to a `.json` file.

        Args:
            fpath: path to the `.json` file where to save the AUPImO result.
                - must have a `.json` extension or no extension (in which case `.json` is added)
                - if the file already exists, a numerical suffix is added to the filename
        """
        fpath = _validate_and_convert_fpath(fpath, extension=".json")
        fpath = duplicate_filename(fpath)
        payload = self.to_dict(keeptensor=False)
        with fpath.open("w") as f:
            json.dump(payload, f, indent=4)

    @classmethod
    def load(cls, fpath: str | Path) -> "AUPImOResult":
        """Load the AUPImO result from a `.json` file.

        Args:
            fpath: path to the `.json` file where to load the AUPImOResult.
        """
        fpath = _validate_and_convert_fpath(fpath, extension=".json")
        with fpath.open("r") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise InvalidAUPImOResult(f"Invalid {cls.__name__} object from file {fpath}, expected a dictionary.")
        try:
            return cls.from_dict(payload)
        except InvalidAUPImOResult as ex:
            raise InvalidAUPImOResult(f"Could not load {cls.__name__} from file {fpath}") from ex


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

    def compute(self) -> tuple[PImOResult, AUPImOResult]:  # type: ignore[override]
        """Compute the Area Under the Per-Image Overlap curves (AUPImO).

        Returns:
            [0] PImOResult (`anomalib.utils.metrics.perimg.pimo.PImOResult`)
            [1] AUPImOResult (`anomalib.utils.metrics.perimg.pimo.AUPImOResult`)
        """

        if self.is_empty:
            raise RuntimeError("Cannot compute AUPImO without any data.")

        pimoresult = super().compute()
        _, __, shared_fpr, tprs, ___ = pimoresult.to_tuple()

        # get the index of the value in `shared_fpr` that is closest to `self.ubound in abs value
        # knwon issue: `shared_fpr[ubound_idx]` might not be exactly `self.ubound`
        # but it's ok because `num_thresholds` should be large enough so that the error is negligible
        ubound_idx = pimoresult.threshold_index_at(self.ubound)

        # limit the curves to the integration range [0, ubound]
        # `shared_fpr` and `tprs` are in descending order; `flip()` reverts to ascending order
        tprs_tocompute_auc: Tensor = tprs[:, ubound_idx:].flip(dims=(1,))
        shared_fpr_tocompute_auc: Tensor = shared_fpr[ubound_idx:].flip(dims=(0,))

        aucs: Tensor = torch.trapezoid(tprs_tocompute_auc, x=shared_fpr_tocompute_auc, dim=1)

        # normalize the size of `aucs` by dividing by the x-range size
        # clip(0, 1) makes sure that the values are in [0, 1] (in case of numerical errors)
        aucs = (aucs / self.ubound).clip(0, 1)

        return pimoresult, AUPImOResult(
            # `MEAN_PERIMAGE_FPR` is the only one implemented for now
            SharedFPRMetric.MEAN_PERIMAGE_FPR,
            SharedFPRScale.LINEAR,
            0.0,
            self.ubound,
            pimoresult.threshold_at(0.0),
            pimoresult.threshold_at(self.ubound),
            aucs,
        )

    def plot_all_pimo_curves(
        self,
        ax: Axes | None = None,
    ) -> tuple[Figure | None, Axes]:
        """Plot shared FPR vs Per-Image Overlap (PImO) curves (all curves).
        Integration range is shown when `self.ubound < 1`.
        """

        if self.is_empty:
            return None, None

        curves, _ = self.compute()
        _, __, shared_fpr, tprs, image_classes = curves.to_tuple()

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
        _, aupimos = self.compute()
        stats = perimg_boxplot_stats(values=aupimos.aucs, image_classes=aupimos.image_classes, only_class=1)
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

        curves, _ = self.compute()
        thresholds, fprs, shared_fpr, tprs, image_classes = curves.to_tuple()
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

        _, aupimos = self.compute()
        fig, ax = plot_aupimo_boxplot(aupimos.aucs, aupimos.image_classes, ax=ax)
        _add_avline_at_score_random_model(ax, 0.5)  # todo correct the random model score for ubound < 1
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

        curves, _ = self.compute()
        thresholds, fprs, shared_fpr, _, image_classes = curves.to_tuple()

        # FRP upper bound is threshold lower bound
        th_lbound = curves.threshold_at(self.ubound)

        plot_th_fpr_curves_norm_only(
            fprs, shared_fpr, thresholds, image_classes, th_lb_fpr_ub=(th_lbound, self.ubound), ax=ax[0]
        )

        plot_pimfpr_curves_norm_only(fprs, shared_fpr, image_classes, ax=ax[1])
        _add_integration_range_to_pimo_curves(ax[1], (None, self.ubound))

        return fig, ax

    # TODO remove this
    def save(self, fpath: str | Path, curve: bool = True):
        raise NotImplementedError("This method was deleted. Use `AUPImOResult.save()` instead.")

    @staticmethod
    def load(fpath: str | Path, curve: bool = True) -> Tensor | tuple[PImOResult, Tensor]:  # type: ignore
        raise NotImplementedError("This method was deleted. Use `AUPImOResult.save()` instead.")


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

    @staticmethod
    def random_model_auc_from_bounds(lbound: float, ubound: float) -> float:
        lbound = _validate_and_convert_rate(lbound)
        ubound = _validate_and_convert_rate(ubound)

        if lbound >= ubound:
            raise ValueError(f"Expected argument `lbound` to be < `ubound`, but got {lbound} >= {ubound}.")

        max_primitive_auc = torch.log(ubound / lbound)
        random_model_primitive_auc = ubound - lbound
        return (random_model_primitive_auc / max_primitive_auc).item()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lbound={self.lbound}, ubound={self.ubound})"

    def compute(self) -> tuple[PImOResult, AUPImOResult]:  # type: ignore[override]
        """Compute the Area Under the Log Per-Image Overlap curves (AULogPImO).

        Returns: (PImOResult, aucs)
            [0] PImOResult (`anomalib.utils.metrics.perimg.pimo.PImOResult`)
            [1] AUPImOResult (`anomalib.utils.metrics.perimg.pimo.AUPImOResult`)
        """

        if self.is_empty:
            raise RuntimeError("Cannot compute AULogPImO without any data.")

        pimoresult = super().compute()
        thresholds, _, shared_fpr, tprs, __ = pimoresult.to_tuple()

        # get the index of the value in `shared_fpr` that is closest to `self.ubound in abs value
        # knwon issue: `shared_fpr[ubound_idx]` might not be exactly `self.ubound`
        # but it's ok because `num_thresholds` should be large enough so that the error is negligible
        ubound_th_idx = pimoresult.threshold_index_at(self.ubound)
        lbound_th_idx = pimoresult.threshold_index_at(self.lbound)

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
        tprs_tocompute_auc: Tensor = tprs[:, ubound_th_idx : (lbound_th_idx + 1)].flip(dims=(1,))
        shared_fpr_tocompute_auc: Tensor = shared_fpr[ubound_th_idx : (lbound_th_idx + 1)].flip(dims=(0,))

        # as described in the class's docstring:
        shared_fpr_tocompute_auc = shared_fpr_tocompute_auc.log()

        # deal with an edge case
        shared_fpr_is_invalid = ~shared_fpr_tocompute_auc.isfinite()

        if shared_fpr_is_invalid.all():
            raise RuntimeError(
                "Cannot compute AULogPImO because the shared fpr integration range is empty. "
                "Try increasing the number of thresholds."
            )
        elif shared_fpr_is_invalid.any():
            # raise a warning
            warnings.warn(
                "Some values in the shared fpr integration range are nan. "
                "The AULogPImO will be computed without these values."
            )

            # get rid of nan values by removing them from the integration range
            shared_fpr_tocompute_auc = shared_fpr_tocompute_auc[~shared_fpr_is_invalid]
            tprs_tocompute_auc = tprs_tocompute_auc[:, ~shared_fpr_is_invalid]

        num_points_in_integration_range = shared_fpr_tocompute_auc.size(0)

        LOW_NUMBER_POINTS_INTEGRATION = 100  # TODO move me to a btter place
        if num_points_in_integration_range <= 30:
            raise RuntimeError(
                "Cannot compute AULogPImO because the shared fpr integration range doesnt have enough points. "
                "Try increasing the number of thresholds."
            )

        elif num_points_in_integration_range < LOW_NUMBER_POINTS_INTEGRATION:
            warnings.warn(
                f"Number of points in the shared fpr integration range is low ({num_points_in_integration_range}). "
                "Try increasing the number of thresholds."
            )

        aucs: Tensor = torch.trapezoid(tprs_tocompute_auc, x=shared_fpr_tocompute_auc, dim=1)

        # normalize, then clip(0, 1) makes sure that the values are in [0, 1] in case of numerical errors
        aucs = (aucs / self.max_primitive_auc).clip(0, 1)

        return pimoresult, AUPImOResult(
            # `MEAN_PERIMAGE_FPR` is the only one implemented for now
            SharedFPRMetric.MEAN_PERIMAGE_FPR,
            SharedFPRScale.LOG,
            self.lbound,
            self.ubound,
            pimoresult.threshold_at(self.lbound),
            pimoresult.threshold_at(self.ubound),
            aucs,
        )

    def plot_all_logpimo_curves(
        self,
        ax: Axes | None = None,
    ) -> tuple[Figure | None, Axes]:
        """Plot log10( shared FPR ) vs Per-Image Overlap (LogPImO) curves (all curves)."""

        if self.is_empty:
            return None, None

        curves, _ = self.compute()
        _, __, shared_fpr, tprs, image_classes = curves.to_tuple()

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
        _, aulogpimos = self.compute()
        stats = perimg_boxplot_stats(values=aulogpimos.aucs, image_classes=aulogpimos.image_classes, only_class=1)
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

        curves, _ = self.compute()
        _, __, shared_fpr, tprs, image_classes = curves.to_tuple()
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

        _, aulogpimos = self.compute()
        fig, ax = plot_aulogpimo_boxplot(aulogpimos.aucs, aulogpimos.image_classes, self.random_model_auc, ax=ax)
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

        curves, __ = self.compute()
        thresholds, fprs, shared_fpr, _, image_classes = curves.to_tuple()

        # FRP upper bound is threshold lower bound
        th_lbound = curves.threshold_at(self.ubound)

        # FPR lower bound is threshold upper bound
        th_ubound = curves.threshold_at(self.lbound)

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

    # TODO remove this
    def save(self, fpath: str | Path, curve: bool = True):
        raise NotImplementedError("Not implemented anymore, use `AUPImOResult.save()` instead.")

    # TODO remove this
    @staticmethod
    def load(fpath: str | Path, curve: bool = True) -> Tensor | tuple[PImOResult, Tensor]:  # type: ignore
        raise NotImplementedError("Not implemented anymore, use `AUPImOResult.save()` instead.")
