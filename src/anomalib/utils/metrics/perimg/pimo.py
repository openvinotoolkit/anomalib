"""Per-Image Overlap (PIMO, pronounced pee-mo) curve.

Two variants of AUCs are implemented:
    - AUPImO: Area Under the Per-Image Overlap (PIMO) curves.
              I.e. a metric of per-image average TPR.

another branch
TODO make it possible to define the threshold lower bound before computing the binclf curves
design: find the lower bound here before calling the super().compute(),
    then pass down the lower bound to the super().compute()
so use this to get a binclf only in the considered th region (below the th lower bound is not visible)

for shared fpr = mean( perimg fpr ) == set fpr
    find the th = fpr^-1( MAX_FPR ) with a binary search on the pixels of the norm images
    i.e. it's not necessary to compute the perimg fpr curves (tf. binclf curves) in advance
for other shared fpr alternatives, it's necessary to compute the perimg fpr curves first anyway

further: also choose the th upper bound to be the max score at normal pixels
"""


from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.pyplot import Figure
from matplotlib.ticker import FixedLocator
from torch import Tensor

from .binclf_curve import PerImageBinClfCurve
from .common import _validate_image_classes, _validate_perimg_rate_curves, _validate_rate_curve

LOG_FPR_EPSILON_ORDER = -5
LOG_FPR_EPSILON = 10**LOG_FPR_EPSILON_ORDER


# =========================================== VALIDATIONS ===========================================


def _validate_atleast_one_anomalous_image(image_classes: Tensor):
    if (image_classes == 1).sum() == 0:
        raise ValueError("Expected argument at least one anomalous image, but found none.")


def _validate_atleast_one_normal_image(image_classes: Tensor):
    if (image_classes == 0).sum() == 0:
        raise ValueError("Expected argument at least one normal image, but found none.")


# =========================================== PLOT ===========================================


def plot_pimo_curves(
    shared_fpr: Tensor,
    tprs: Tensor,
    image_classes: Tensor,
    ax: Axes | None = None,
    logfpr: bool = False,
    *kwargs_perimg,
    **kwargs_shared,
) -> tuple[Figure | None, Axes]:
    """Plot shared FPR vs Per-Image Overlap (PImO) curves.

    The `image_classes` tensor is used to filter out the normal images, while making it possible to
        keep the indices of the anomalous images.

    Args:
        shared_fpr: shape (num_thresholds,)
        tprs: shape (num_images, num_thresholds)
        image_classes: shape (num_images,)
        ax: matplotlib Axes
        logfpr: whether to use log scale for the FPR axis

        *kwargs_perimg: keyword arguments passed to `ax.plot()` and SPECIFIC to each curve
                            if provided it should be a list of dicts of length `num_images`
        **kwargs: keyword arguments passed to `ax.plot()` and SHARED by all curves

        If both `kwargs_perimg` and `kwargs_shared` have the same key, the value in `kwargs_perimg` will be used.

    Returns:
        fig, ax
    """

    _validate_perimg_rate_curves(tprs, nan_allowed=True)  # normal images have `nan`s
    _validate_rate_curve(shared_fpr)
    _validate_image_classes(image_classes)

    # `shared_fpr` and `tprs` have the same number of thresholds
    if tprs.shape[1] != shared_fpr.shape[0]:
        raise ValueError(
            f"Expected argument `tprs` to have the same number of thresholds as argument `shared_fpr`, "
            f"but got {tprs.shape[1]} thresholds and {shared_fpr.shape[0]} thresholds, respectively."
        )

    # `tprs` and `image_classes` have the same number of images
    if image_classes is not None:
        if tprs.shape[0] != image_classes.shape[0]:
            raise ValueError(
                f"Expected argument `tprs` to have the same number of images as argument `image_classes`, "
                f"but got {tprs.shape[0]} images and {image_classes.shape[0]} images, respectively."
            )

    # specific to TPR curves
    _validate_atleast_one_anomalous_image(image_classes)
    # there may be `nan`s but only in the normal images
    # in the curves of anomalous images, there should NOT be `nan`s
    _validate_perimg_rate_curves(tprs[image_classes == 1], nan_allowed=False)

    if len(kwargs_perimg) == 0:
        pass

    # `kwargs_perimg` must have `num_images` dicts
    elif len(kwargs_perimg) != tprs.shape[0]:
        raise ValueError(
            f"Expected argument `kwargs_perimg` to have the same number of dicts as number of images in `tprs`, "
            f"but got {len(kwargs_perimg)} dicts and {tprs.shape[0]} images, respectively."
        )

    elif not all(isinstance(kwargs, dict) for kwargs in kwargs_perimg):
        raise ValueError("Expected argument `kwargs_perimg` to be a list of dicts, but got other type(s).")

    fig, ax = plt.subplots() if ax is None else (None, ax)

    # override defaults with user-provided values
    kwargs_shared = {
        **dict(linewidth=1, linestyle="-", alpha=0.3),
        **kwargs_shared,
    }

    for idx, curve in enumerate(tprs):
        img_cls = image_classes[idx]
        if img_cls == 0:  # normal image
            continue
        kwargs_specific = kwargs_perimg[idx] if len(kwargs_perimg) > 0 else {}
        kw = {**kwargs_shared, **kwargs_specific}
        ax.plot(shared_fpr, curve, label=f"idx={idx:03}", **kw)

    ax.set_xlabel("Shared FPR")

    if logfpr:
        ax.set_xscale("log")
        ax.set_xlim(LOG_FPR_EPSILON, 1)
        ticks_major = np.logspace(LOG_FPR_EPSILON_ORDER, 0, abs(LOG_FPR_EPSILON_ORDER) + 1)
        formatter_major = lambda val, pos: f"{val:.0e}"  # noqa: E731
        ticks_minor = np.logspace(LOG_FPR_EPSILON_ORDER, 0, 2 * abs(LOG_FPR_EPSILON_ORDER) + 1)

    else:
        XLIM_EPSILON = 0.01
        ax.set_xlim(0 - XLIM_EPSILON, 1 + XLIM_EPSILON)
        ticks_major = np.linspace(0, 1, 6)
        formatter_major = lambda val, pos: f"{val:.0%}"  # noqa: E731
        ticks_minor = np.linspace(0, 1, 11)

    ax.xaxis.set_major_locator(FixedLocator(ticks_major))
    ax.xaxis.set_major_formatter(formatter_major)
    ax.xaxis.set_minor_locator(FixedLocator(ticks_minor))

    ax.set_ylabel("Per-Image Overlap (in-image TPR)")
    YLIM_EPSILON = 0.01
    ax.set_ylim(0 - YLIM_EPSILON, 1 + YLIM_EPSILON)
    ax.yaxis.set_major_locator(FixedLocator(np.linspace(0, 1, 6)))
    ax.yaxis.set_major_formatter(lambda val, pos: f"{val:.0%}")
    ax.yaxis.set_minor_locator(FixedLocator(np.linspace(0, 1, 11)))

    ax.set_title("Per-Image Overlap Curves")

    return fig, ax


# =========================================== METRICS ===========================================


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
        Rationale: this will further punish models that have outlier-ly FPs in normal images.

    FP: False Positive
    FPR: False Positive Rate
    TP: True Positive
    TPR: True Positive Rate
    """

    def compute(self) -> tuple[Tensor, Tensor, Tensor]:  # type: ignore
        """Compute the PImO curve.


        Returns:
            thresholds: shape (num_thresholds,), dtype as given in update()
            shared_fpr: shape (num_thresholds,), dtype float64, \in [0, 1]
            tprs: shape (num_images, num_thresholds), dtype float64,
                \in [0, 1] for anomalous images, `nan` for normal images

            `num_thresholds` is an attribute of the parent class.
            `num_images` depends on the data seen by the model at the update() calls.

        """

        # shape: (num_images, num_thresholds, 2, 2), where (2, 2) is (true class, predicted class)
        thresholds, binclf_curves = super().compute()

        image_class = self._image_classes_tensor
        _validate_atleast_one_normal_image(image_class)  # necessary for the FPR
        _validate_atleast_one_anomalous_image(image_class)  # necessary for the TPR

        # the next tensors have shape (num_images, num_thresholds)
        tps = binclf_curves[..., 1, 1]
        pos = binclf_curves[..., 1, :].sum(dim=-1)

        # tprs will be nan if pos == 0 (normal image), which is expected
        tprs = tps.to(torch.float64) / pos.to(torch.float64)

        # the next tensors have shape (num_images, num_thresholds)
        fps = binclf_curves[..., 0, 1]
        neg = binclf_curves[..., 0, :].sum(dim=-1)

        # it can be `nan` if an anomalous image is fully covered by the mask
        # but it's ok because we will use only the normal images
        fprs = fps.to(torch.float64) / neg.to(torch.float64)

        # see note about shared FPR alternatives in the class's docstring
        shared_fpr = fprs[image_class == 0].mean(dim=0)  # shape: (num_thresholds,)

        return thresholds, shared_fpr, tprs

    def plot(
        self,
        logfpr: bool = False,
        ax: Axes | None = None,
    ) -> tuple[Figure | None, Axes]:
        """Plot shared FPR vs Per-Image Overlap (PImO) curves.

        Args:
            logfpr: whether to use log scale for the FPR axis (X-axis)

        Returns:
            fig, ax
        """
        _, shared_fpr, tprs = self.compute()
        fig, ax = plot_pimo_curves(
            shared_fpr=shared_fpr,
            tprs=tprs,
            image_classes=self._image_classes_tensor,
            ax=ax,
            logfpr=logfpr,
        )
        ax.set_xlabel("Mean FPR on Normal Images")
        return fig, ax


class AUPImO(PImO):
    """Area Under the Per-Image Overlap (PImO) curve.

    AU is computed by the trapezoidal rule, each curve being treated separately.

    TODO get lower bound from shared fpr and only compute the area under the curve in the considered region
        --> (1) add class attribute, (2) add integration range at plot, (3) filter curves before intergration
    """

    def compute(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:  # type: ignore
        """Compute the Area Under the Per-Image Overlap curves (AUPImO).

        Returns:
            (thresholds, shared_fpr, tprs, aucs)
            (thresholds, shared_fpr, tprs): same as in PImO.compute()
            aucs: shape (num_images,), dtype float64, \in [0, 1]
        """

        thresholds: Tensor  # (num_thresholds,)
        shared_fpr: Tensor  # (num_thresholds,)
        tprs: Tensor  # (num_images, num_thresholds)
        thresholds, shared_fpr, tprs = super().compute()

        # TODO find lower bound from shared fpr

        # `shared_fpr` and `tprs` are in descending order; revert to ascending order
        aucs: Tensor = torch.trapezoid(tprs.flip(dims=(1,)), x=shared_fpr.flip(dims=(0,)), dim=1)

        return thresholds, shared_fpr, tprs, aucs

    def plot_pimo_curves(
        self,
        ax: Axes | None = None,
    ) -> tuple[Figure | None, Axes]:
        """Plot shared FPR vs Per-Image Overlap (PImO) curves."""
        thresholds, shared_fpr, tprs, aucs = self.compute()
        # TODO customize special cases
        fig, ax = plot_pimo_curves(
            shared_fpr=shared_fpr,
            tprs=tprs,
            image_classes=self._image_classes_tensor,
            ax=ax,
            logfpr=False,
        )
        ax.set_xlabel("Mean FPR on Normal Images")
        return fig, ax


class AULogPImO(PImO):
    """Area Under the Per-Image Overlap (PIMO, pronounced pee-mo) curves with log(FPR) (instead of FPR) in the X-axis

    This will further give more importance/visibility to the low FPR region.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError("**coming up later**")
