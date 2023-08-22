"""  TODO
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy
import pandas as pd
import scipy
import torch
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.pyplot import Figure
from matplotlib.ticker import FixedLocator, IndexLocator, PercentFormatter
from torch import Tensor

from .common import (
    _validate_and_convert_aucs,
    _validate_and_convert_models_dict,
    _validate_and_convert_rate,
    _validate_and_convert_threshold,
    _validate_atleast_one_anomalous_image,
    _validate_atleast_one_normal_image,
    _validate_image_class,
    _validate_image_classes,
    _validate_perimg_rate_curves,
    _validate_rate_curve,
    _validate_thresholds,
    perimg_boxplot_stats,
)

# =========================================== FORMAT ===========================================


def _format_axis_rate_metric_linear(
    ax: Axes, axis: int, lims: tuple[float, float] = (0.0, 1.0), num_ticks_major: int = 6
) -> None:
    """Format an axis for a rate metric plot.

    Args:
        ax (Axes): Axis to format.
        axis (int): Axis to format. Must be 0 (X-axis) or 1 (Y-axis).
        lims_epsilon (float): Epsilon to add to the axis limits. Defaults to 0.01.
    """

    if not isinstance(ax, Axes) or not isinstance(axis, int):
        raise ValueError("Expected arguments `ax` to be an Axes and `axis` to be an integer.")

    # add a small margin to the limits
    lims_size = lims[1] - lims[0]
    lims_epsilon = 0.01 * lims_size
    lims_with_epsilon = (lims[0] - lims_epsilon, lims[1] + lims_epsilon)

    ticks_major = numpy.linspace(*lims, num_ticks_major)
    formatter_major = PercentFormatter(1, decimals=0)
    # twice as many major ticks
    ticks_minor = numpy.linspace(*lims, (num_ticks_major - 1) * 2 + 1)

    if axis == 0:
        ax.set_xlim(*lims_with_epsilon)
        ax.xaxis.set_major_locator(FixedLocator(ticks_major))
        ax.xaxis.set_major_formatter(formatter_major)
        ax.xaxis.set_minor_locator(FixedLocator(ticks_minor))

    elif axis == 1:
        ax.set_ylim(*lims_with_epsilon)
        ax.yaxis.set_major_locator(FixedLocator(ticks_major))
        ax.yaxis.set_major_formatter(formatter_major)
        ax.yaxis.set_minor_locator(FixedLocator(ticks_minor))

    else:
        raise ValueError(f"`axis` must be 0 (X-axis) or 1 (Y-axis), but got {axis}.")


def _format_axis_rate_metric_log(ax: Axes, axis: int, lower_lim: float = 1e-3, upper_lim: float = 1) -> None:
    if not isinstance(ax, Axes) or not isinstance(axis, int):
        raise ValueError("Expected arguments `ax` to be an Axes and `axis` to be an integer.")

    assert lower_lim > 0, f"Expected argument `lower_lim` to be positive, but got {lower_lim}."
    assert (
        upper_lim > lower_lim
    ), f"Expected `upper_lim` > `lower_lim`, but got {upper_lim} and {lower_lim}, respectively."

    lims = (lower_lim, upper_lim)

    lims = (lower_lim, upper_lim)
    lower_lim_rounded_exponent = int(numpy.floor(numpy.log10(lower_lim)))
    upper_lim_rounded_exponent = int(numpy.ceil(numpy.log10(upper_lim)))
    num_exponents = upper_lim_rounded_exponent - lower_lim_rounded_exponent + 1

    ticks_major = numpy.logspace(lower_lim_rounded_exponent, upper_lim_rounded_exponent, num_exponents)

    lims_epsilon = 0.1
    lims = (
        (10**lower_lim_rounded_exponent) * (1 - lims_epsilon),
        (10**upper_lim_rounded_exponent) * (1 + lims_epsilon),
    )

    def formatter_major(x, pos):
        return f"{100 * x}%" if x < 0.01 else f"{100 * x:.0f}%"

    ticks_minor = numpy.logspace(lower_lim_rounded_exponent, upper_lim_rounded_exponent, 3 * (num_exponents - 1) + 1)

    if axis == 0:
        ax.set_xscale("log")
        ax.set_xlim(*lims)
        ax.xaxis.set_major_locator(FixedLocator(ticks_major))
        ax.xaxis.set_major_formatter(formatter_major)
        ax.xaxis.set_minor_locator(FixedLocator(ticks_minor))

    elif axis == 1:
        ax.set_yscale("log")
        ax.set_ylim(*lims)
        ax.yaxis.set_major_locator(FixedLocator(ticks_major))
        ax.yaxis.set_major_formatter(formatter_major)
        ax.yaxis.set_minor_locator(FixedLocator(ticks_minor))

    else:
        raise ValueError(f"`axis` must be 0 (X-axis) or 1 (Y-axis), but got {axis}.")


def _bounded_lims(
    ax: Axes, axis: int, bounds: tuple[float | None, float | None] = (None, None), lims_epsilon: float = 0.01
):
    """Snap X/Y-axis limits to stay within the given bounds."""

    assert len(bounds) == 2, f"Expected argument `bounds` to be a tuple of size 2, but got size {len(bounds)}."
    bounds = (
        None if bounds[0] is None else bounds[0] - lims_epsilon,
        None if bounds[1] is None else bounds[1] + lims_epsilon,
    )

    if axis == 0:
        lims = ax.get_xlim()
    elif axis == 1:
        lims = ax.get_ylim()
    else:
        raise ValueError(f"Unknown axis {axis}. Must be 0 (X-axis) or 1 (Y-axis).")

    newlims = list(lims)

    if bounds[0] is not None and lims[0] < bounds[0]:
        newlims[0] = bounds[0]

    if bounds[1] is not None and lims[1] > bounds[1]:
        newlims[1] = bounds[1]

    if axis == 0:
        ax.set_xlim(newlims)
    else:
        ax.set_ylim(newlims)


def _format_axis_imgidx(ax, num_imgs: int):
    assert num_imgs > 0
    ax.set_xlabel("Image Index")
    EPSILON = 0.01
    ax.set_xlim(0 - EPSILON * num_imgs, num_imgs - 1 + EPSILON * num_imgs)
    ax.xaxis.set_major_locator(IndexLocator(5, 0))
    ax.xaxis.set_minor_locator(IndexLocator(1, 0))
    ax.grid(axis="x", which="major")
    ax.grid(axis="x", which="minor", linestyle="--", alpha=0.3)


# =========================================== GENERIC ===========================================


def _plot_perimg_metric_boxplot(
    ax: Axes,
    values: Tensor,
    image_classes: Tensor,
    bp_stats: list,
    only_class: int | None = None,
):
    _validate_image_classes(image_classes)
    _validate_image_class(only_class)

    if values.ndim != 1:
        raise ValueError(f"Expected argument `values` to be a 1D tensor, but got {values.ndim}D tensor.")

    if values.shape != image_classes.shape:
        raise ValueError(
            "Expected arguments `values` and `image_classes` to have the same shape, "
            f"but got {values.shape} and {image_classes.shape}."
        )

    if only_class is not None and only_class not in image_classes:
        raise ValueError(f"Argument `only_class` is {only_class}, but `image_classes` does not contain this class.")

    # only consider images of the given class
    imgs_mask = (
        torch.ones_like(image_classes, dtype=torch.bool) if only_class is None else (image_classes == only_class)
    )
    imgs_idxs = torch.nonzero(imgs_mask).squeeze(1)

    ax.boxplot(
        values[imgs_mask],
        vert=False,
        widths=0.5,
        showmeans=True,
        showcaps=True,
        notch=False,
    )
    _ = ax.set_yticks([])

    num_images = len(imgs_idxs)
    num_flierlo = len([s for s in bp_stats if s["statistic"] == "flierlo" and s["imgidx"] in imgs_idxs])
    num_flierhi = len([s for s in bp_stats if s["statistic"] == "flierhi" and s["imgidx"] in imgs_idxs])

    ax.annotate(
        text=f"Number of images\n    total: {num_images}\n    fliers: {num_flierlo} low, {num_flierhi} high",
        xy=(0.03, 0.95),
        xycoords="axes fraction",
        xytext=(0, 0),
        textcoords="offset points",
        annotation_clip=False,
        verticalalignment="top",
    )


def _plot_perimg_curves(
    ax: Axes,
    x: Tensor,
    ys: Tensor,
    *kwargs_perimg: dict[str, Any | None] | None,
    **kwargs_shared,
) -> Axes:
    """
    Args:
        ax: matplotlib Axes
        x: shape (n,)
        ys: shape (num_curves, n)
        *kwargs_perimg: keyword arguments passed to `ax.plot()` and SPECIFIC to each curve
                            a sequence of objects of length `num_curves`.
                            If None, that curve will not be ploted.
                            Otherwise, it should be a dict of keyword arguments passed to `ax.plot()`.

        **kwargs_shared: keyword arguments passed to `ax.plot()` and SHARED by all curves

        If both `kwargs_perimg` and `kwargs_shared` have the same key, the value in `kwargs_perimg` will be used.
    """

    if not isinstance(x, Tensor) or not isinstance(ys, Tensor):
        raise TypeError(
            f"Expected arguments `x` and `ys` to be tensors, but got {type(x)} and {type(ys)}, respectively."
        )

    if x.ndim != 1 or ys.ndim != 2:
        raise ValueError(
            "Expected arguments `x` and `ys` to be 1 amd 2-dimensional tensors,"
            f" but got {x.ndim}D and {ys.ndim}D tensors respectively."
        )

    if x.shape[0] != ys.shape[1]:
        raise ValueError(
            "Expected argument `ys.shape[1]` to be equal to `x.shape[0]`, "
            f"but got {ys.shape[1]} and {x.shape[0]}, respectively."
        )

    num_curves = ys.shape[0]
    num_kwargs_perimg = len(kwargs_perimg)

    if num_kwargs_perimg != num_curves:
        raise ValueError(
            "Expected the number of keyword arguments to be equal to the number of curves, "
            f"but got {num_kwargs_perimg} and {num_curves}, respectively."
        )

    othertypes = {type(kws) for kws in kwargs_perimg if kws is not None and not isinstance(kws, dict)}

    if len(othertypes) > 0:
        raise ValueError(
            "Expected arguments `kwargs_perimg` to be a dict or None, "
            f"but found {sorted(othertypes, key=lambda t: t.__name__)} instead."
        )

    for y, kwargs_specific in zip(ys, kwargs_perimg):
        if kwargs_specific is None:
            continue  # skip this curve

        # override the shared kwargs with the image-specific kwargs
        kw: dict[str, Any] = {**kwargs_shared, **kwargs_specific}
        ax.plot(x, y, **kw)


# =========================================== BOXPLOTS ===========================================


def plot_aupimo_boxplot(
    aucs: Tensor,
    image_classes: Tensor,
    ax: Axes | None = None,
) -> tuple[Figure | None, Axes]:
    _validate_and_convert_aucs(aucs, nan_allowed=True)
    _validate_atleast_one_anomalous_image(image_classes)

    fig, ax = plt.subplots() if ax is None else (None, ax)

    bp_stats = perimg_boxplot_stats(aucs, image_classes, only_class=1)

    _plot_perimg_metric_boxplot(
        ax=ax,
        values=aucs,
        image_classes=image_classes,
        only_class=1,
        bp_stats=bp_stats,
    )

    # don't go beyond the [0, 1]
    _bounded_lims(ax, axis=0, bounds=(0, 1))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("AUPImO [%]")
    ax.set_title("Area Under the Per-Image Overlap (AUPImO) Boxplot")

    return fig, ax


def plot_aulogpimo_boxplot(
    aucs: Tensor,
    image_classes: Tensor,
    random_model_auc: float | None = None,
    ax: Axes | None = None,
) -> tuple[Figure | None, Axes]:
    if random_model_auc is not None:
        _validate_and_convert_rate(random_model_auc, nonzero=True, nonone=True)
    fig, ax = plot_aupimo_boxplot(aucs, image_classes, ax=ax)
    ax.set_xlabel("AULogPImO [%]")
    ax.set_title("Area Under the Log Per-Image Overlap (AULogPImO) Boxplot")
    if random_model_auc is not None:
        _add_avline_at_score_random_model(ax, random_model_auc)
    return fig, ax


# =========================================== PImO ===========================================


def plot_all_pimo_curves(
    shared_fpr: Tensor,
    tprs: Tensor,
    image_classes: Tensor,
    ax: Axes | None = None,
) -> tuple[Figure | None, Axes]:
    """Plot shared FPR vs Per-Image Overlap (PImO) curves (same as in-image TPR).

    Args:
        ax: matplotlib Axes
        shared_fpr: shape (num_thresholds,)
        tprs: shape (num_images, num_thresholds)
        image_classes: shape (num_images,)
            The `image_classes` tensor is used to filter out the normal images, while making it possible to
            keep the indices of the anomalous images.
    Returns:
        fig, ax
    """
    # ** validate **
    _validate_rate_curve(shared_fpr)
    _validate_perimg_rate_curves(tprs, nan_allowed=True)  # normal images have `nan`s
    _validate_image_classes(image_classes)

    if tprs.shape[0] != image_classes.shape[0]:
        raise ValueError(
            f"Expected argument `tprs` to have the same number of images as argument `image_classes`, "
            f"but got {tprs.shape[0]} images and {image_classes.shape[0]} images, respectively."
        )

    _validate_atleast_one_anomalous_image(image_classes)
    # there may be `nan`s but only in the normal images
    # in the curves of anomalous images, there should NOT be `nan`s
    _validate_perimg_rate_curves(tprs[image_classes == 1], nan_allowed=False)

    # ** plot **
    fig, ax = plt.subplots(figsize=(7, 6)) if ax is None else (None, ax)

    _plot_perimg_curves(
        ax,
        shared_fpr,
        tprs,
        *[
            ({"label": f"idx={imgidx:03}"} if img_cls == 1 else None)  # a generic label  # don't plot this curve
            for imgidx, img_cls in enumerate(image_classes)
        ],  # type: ignore
        # shared kwargs
        alpha=0.3,
    )

    # ** format **
    _format_axis_rate_metric_linear(ax, axis=0)
    ax.set_xlabel("Shared FPR")
    _format_axis_rate_metric_linear(ax, axis=1)
    ax.set_ylabel("Per-Image Overlap (in-image TPR)")
    ax.set_title("Per-Image Overlap (PImO) Curves")

    return fig, ax


def plot_boxplot_pimo_curves(
    shared_fpr,
    tprs,
    image_classes,
    bp_stats: list[dict[str, str | int | float | None]],
    ax: Axes | None = None,
) -> tuple[Figure | None, Axes]:
    """Plot shared FPR vs Per-Image Overlap (PImO) curves only for the boxplot stats cases.

    Args:
        ax: matplotlib Axes
        shared_fpr: shape (num_thresholds,)
        tprs: shape (num_images, num_thresholds)
        image_classes: shape (num_images,)
            The `image_classes` tensor is used to filter out the normal images, while making it possible to
            keep the indices of the anomalous images.

        bp_stats: list of dicts, each dict is a boxplot stat of AUPImO values
                  refer to `anomalib.utils.metrics.perimg.common.perimg_boxplot_stats()`

    Returns:
        fig, ax
    """

    # ** validate **
    _validate_rate_curve(shared_fpr)
    _validate_perimg_rate_curves(tprs, nan_allowed=True)  # normal images have `nan`s
    _validate_image_classes(image_classes)

    if tprs.shape[0] != image_classes.shape[0]:
        raise ValueError(
            f"Expected argument `tprs` to have the same number of images as argument `image_classes`, "
            f"but got {tprs.shape[0]} images and {image_classes.shape[0]} images, respectively."
        )

    _validate_atleast_one_anomalous_image(image_classes)
    # there may be `nan`s but only in the normal images
    # in the curves of anomalous images, there should NOT be `nan`s
    _validate_perimg_rate_curves(tprs[image_classes == 1], nan_allowed=False)

    if len(bp_stats) == 0:
        raise ValueError("Expected argument `bp_stats` to have at least one dict, but got none.")

    # ** kwargs_perimg **

    # it is sorted so that only the first one has a label (others are plotted but don't show in the legend)
    imgidxs_toplot_fliers: list[int] = sorted(
        {s["imgidx"] for s in bp_stats if s["statistic"] in ("flierlo", "flierhi")}  # type: ignore
    )
    imgidxs_toplot_others = {s["imgidx"] for s in bp_stats if s["statistic"] not in ("flierlo", "flierhi")}

    kwargs_perimg = []
    num_images = len(image_classes)

    for imgidx in range(num_images):
        if imgidx in imgidxs_toplot_fliers:
            kw = dict(linewidth=0.5, color="gray", alpha=0.8, linestyle="--")

            # only one of them will show in the legend
            if imgidx == imgidxs_toplot_fliers[0]:
                kw["label"] = "flier"
            else:
                kw["label"] = None

            kwargs_perimg.append(kw)

            continue

        if imgidx not in imgidxs_toplot_others:
            # don't plot this curve
            kwargs_perimg.append(None)  # type: ignore
            continue

        imgidx_stats = [s for s in bp_stats if s["imgidx"] == imgidx]
        stat_dict = imgidx_stats[0]

        # edge case where more than one stat falls on the same image
        if len(imgidx_stats) > 1:
            stat_dict["statistic"] = " & ".join(s["statistic"] for s in imgidx_stats)  # type: ignore

        stat, nearest = stat_dict["statistic"], stat_dict["nearest"]
        kwargs_perimg.append(dict(label=f"{stat} (AUPImO={nearest:.1%}) (imgidx={imgidx})"))

    # ** plot **

    fig, ax = plt.subplots(figsize=(7, 6)) if ax is None else (None, ax)

    _plot_perimg_curves(ax, shared_fpr, tprs, *kwargs_perimg)

    # ** legend **

    def _sort_legend(handles: list, labels: list[str]):
        """sort the legend by label and put 'flier' at the bottom
        it makes the legend 'more deterministic'
        """

        # [(handle0, label0), (handle1, label1),...]
        handles_labels = list(zip(handles, labels))
        handles_labels = sorted(handles_labels, key=lambda tup: tup[1])

        # ([handle0, handle1, ...], [label0, label1, ...])
        handles, labels = tuple(map(list, zip(*handles_labels)))  # type: ignore

        # put flier at the last position
        if "flier" in labels:
            idx = labels.index("flier")
            handles.append(handles.pop(idx))
            labels.append(labels.pop(idx))

        return handles, labels

    ax.legend(
        *_sort_legend(*ax.get_legend_handles_labels()),
        title="Boxplot Stats",
        loc="lower right",
        fontsize="small",
        title_fontsize="small",
    )

    # ** format **

    _format_axis_rate_metric_linear(ax, axis=0)
    ax.set_xlabel("Shared FPR")
    _format_axis_rate_metric_linear(ax, axis=1)
    ax.set_ylabel("Per-Image Overlap (in-image TPR)")
    ax.set_title("Per-Image Overlap (PImO) Curves (AUC boxplot statistics)")

    return fig, ax


def plot_boxplot_logpimo_curves(
    shared_fpr,
    tprs,
    image_classes,
    bp_stats: list[dict[str, str | int | float | None]],
    lbound: float,
    ubound: float,
    ax: Axes | None = None,
) -> tuple[Figure | None, Axes]:
    """Plot shared FPR vs Log Per-Image Overlap (LogPImO) curves only for the boxplot stats cases.

    Args:
        ax: matplotlib Axes
        shared_fpr: shape (num_thresholds,)
        tprs: shape (num_images, num_thresholds)
        image_classes: shape (num_images,)
            The `image_classes` tensor is used to filter out the normal images, while making it possible to
            keep the indices of the anomalous images.

        bp_stats: list of dicts, each dict is a boxplot stat of AULogPImO values
                  refer to `anomalib.utils.metrics.perimg.common.perimg_boxplot_stats()`

    Returns:
        fig, ax
    """
    lbound = _validate_and_convert_rate(lbound)
    ubound = _validate_and_convert_rate(ubound)

    if lbound >= ubound:
        raise ValueError(f"Expected `lbound` < `ubound`, but got {lbound} and {ubound}, respectively.")

    # other args are validated in `plot_boxplot_pimo_curves()`
    fig, ax = plot_boxplot_pimo_curves(
        shared_fpr,
        tprs,
        image_classes,
        bp_stats,
        ax=ax,
    )
    ax.set_xlabel("Log10 of Shared FPR")
    ax.set_title("Log Per-Image Overlap (LogPImO) Curves (AUC boxplot statistics)")
    _format_axis_rate_metric_log(ax, axis=0, lower_lim=lbound, upper_lim=ubound)
    # they are not exactly the same as the input because the function above rounds them
    xtickmin, xtickmax = ax.xaxis.get_ticklocs()[[0, -1]]
    _add_integration_range_to_pimo_curves(ax, (lbound, ubound), span=(xtickmin < lbound or xtickmax > ubound))
    return fig, ax


def _add_integration_range_to_pimo_curves(
    ax: Axes,
    bounds: tuple[float | None, float | None] = (None, None),
    span: bool = True,
) -> None:
    """Add a vertical span and two vertical lines to the plot to indicate the integration range.

    Args:
        ax: matplotlib Axes
        bounds: (lbound, ubound) where both are floats in [0, 1]
                when None, the corresponding vertical line will not be added
        span: whether to add a vertical span
    """
    current_legend = ax.get_legend()

    if len(bounds) != 2:
        raise ValueError(f"Expected argument `bounds` to be a tuple of size 2, but got size {len(bounds)}.")

    lbound, ubound = bounds

    if lbound is not None:
        _validate_and_convert_rate(lbound)

    if ubound is not None:
        _validate_and_convert_rate(ubound)

    if lbound is not None and ubound is not None and lbound >= ubound:
        raise ValueError(
            f"Expected argument `bounds` to be (lbound, ubound), such that lbound < ubound, but got {bounds}."
        )

    handles = []

    if span:
        handles.append(
            ax.axvspan(
                lbound if lbound is not None else 0,
                ubound if ubound is not None else 1,
                label="FPR Interval",
                color="cyan",
                alpha=0.2,
            )
        )

    def bound2str(bound: float) -> str:
        return f"{100 * bound}%" if bound < 0.01 else f"{100 * bound:.0f}%"

    if ubound is not None:
        handles.append(
            ax.axvline(
                ubound,
                label=f"Upper Bound ({bound2str(ubound)})",
                linestyle="--",
                color="black",
            )
        )

    if lbound is not None:
        handles.append(
            ax.axvline(
                lbound,
                label=f"Lower Bound ({bound2str(lbound)})",
                linestyle="--",
                color="gray",
            )
        )

    ax.legend(
        handles,
        [h.get_label() for h in handles],
        title="AUC integration",
        loc="center right",
        fontsize="small",
        title_fontsize="small",
    )

    if current_legend is not None:
        ax.add_artist(current_legend)


def _add_avline_at_score_random_model(ax: Axes, score: float, axis: int = 1, add_legend: bool = True) -> None:
    """Add a horizontal or vertical line at the score of the random model.
    `axis=0` means X-axis, so horizontal line.
    `axis=1` means Y-axis, so vertical line.
    `axis=1` is the default for backward compatibility.
    """

    if not isinstance(ax, Axes) or not isinstance(axis, int):
        raise ValueError("Expected arguments `ax` to be an Axes and `axis` to be an integer.")

    current_legend = ax.get_legend()

    def auc2str(bound: float) -> str:
        return f"{bound:.1%}"

    if axis == 0:
        handle = ax.axhline(
            score,
            label=f"Random Model ({auc2str(score)})",
            linestyle="--",
            color="red",
        )

    elif axis == 1:
        handle = ax.axvline(
            score,
            label=f"Random Model ({auc2str(score)})",
            linestyle="--",
            color="red",
        )
    else:
        raise ValueError(f"`axis` must be 0 (X-axis, horizontal line) or 1 (Y-axis, vertical line), but got {axis}.")

    if not add_legend:
        return

    ax.legend(
        [handle],
        [handle.get_label()],
        loc="lower right",
        fontsize="small",
    )

    if current_legend is not None:
        ax.add_artist(current_legend)


# =========================================== PImFPR ===========================================


def plot_pimfpr_curves_norm_vs_anom(
    fprs: Tensor,
    shared_fpr: Tensor,
    image_classes: Tensor,
    ax: Axes | None = None,
) -> tuple[Figure | None, Axes]:
    """Plot shared FPR vs in-image FPR curves, normal image (in blue) VS anomalous images (in red).

    Args:
        ax: matplotlib Axes
        shared_fpr: shape (num_thresholds,)
        fprs: shape (num_images, num_thresholds)
        image_classes: shape (num_images,)
    """

    # ** validate ** [GENERIC]
    _validate_rate_curve(shared_fpr)
    _validate_perimg_rate_curves(fprs, nan_allowed=True)  # anomalous images may have `nan`s if all pixels are anomalous
    _validate_image_classes(image_classes)

    if fprs.shape[0] != image_classes.shape[0]:
        raise ValueError(
            f"Expected argument `fprs` to have the same number of images as argument `image_classes`, "
            f"but got {fprs.shape[0]} images and {image_classes.shape[0]} images, respectively."
        )

    # there may be `nan`s but only in the anomalous images
    # in the curves of normal images, there should NOT be `nan`s
    if (image_classes == 0).any():
        _validate_perimg_rate_curves(fprs[image_classes == 0], nan_allowed=False)

    # ** validate ** [SPECIFIC]
    # it's a normal vs. anomalous plot, so there should be at least one of each
    _validate_atleast_one_anomalous_image(image_classes)
    _validate_atleast_one_normal_image(image_classes)

    fig, ax = plt.subplots(figsize=(7, 7)) if ax is None else (None, ax)

    # ** plot **
    kwargs_perimg = [
        dict(
            # color the lines by the image class; normal = blue, anomalous = red
            color="blue" if img_cls == 0 else "red",
            # make the legend only show one normal and one anomalous line by passing `label=None`
            # and just modifying the first of each class in the lines below
            label=None,
        )
        for imgidx, img_cls in enumerate(image_classes)
    ]
    # `[0][0]`: first `[0]` is for the tuple from `numpy.where()`, second `[0]` is for the first index
    kwargs_perimg[numpy.where(image_classes == 0)[0][0]]["label"] = "Normal (blue)"
    kwargs_perimg[numpy.where(image_classes == 1)[0][0]]["label"] = "Anomalous (red)"

    _plot_perimg_curves(
        ax,
        shared_fpr,
        fprs,
        *kwargs_perimg,
        # shared kwargs
        alpha=0.3,
    )

    # ** format **
    _format_axis_rate_metric_linear(ax, axis=0)
    ax.set_xlabel("Shared FPR")
    _format_axis_rate_metric_linear(ax, axis=1)
    ax.set_ylabel("In-Image FPR")
    ax.set_title("FPR: Shared vs In-Image Curves (Norm. vs Anom. Images)")
    ax.legend(loc="lower right", fontsize="small", title_fontsize="small", title="Image Class")
    # TODO change alpha of the legend's handles

    return fig, ax


def plot_pimfpr_curves_norm_only(
    fprs: Tensor,
    shared_fpr: Tensor,
    image_classes: Tensor,
    ax: Axes | None = None,
) -> tuple[Figure | None, Axes]:
    """Plot shared FPR vs in-image FPR curves, only from normal images along with their statistics across the images.

    The statistics curves corresponds to taking (for ex) the mean along the y axis at a given x value in the plot.
    Statistics: min(), max(), and mean() wiht 3 SEM interval.

    Args:
        ax: matplotlib Axes
        shared_fpr: shape (num_thresholds,)
        fprs: shape (num_images, num_thresholds)
        image_classes: shape (num_images,)
    """

    # ** validate **
    _validate_rate_curve(shared_fpr)
    _validate_perimg_rate_curves(fprs, nan_allowed=True)  # anomalous images may have `nan`s if all pixels are anomalous
    _validate_image_classes(image_classes)

    if fprs.shape[0] != image_classes.shape[0]:
        raise ValueError(
            f"Expected argument `fprs` to have the same number of images as argument `image_classes`, "
            f"but got {fprs.shape[0]} images and {image_classes.shape[0]} images, respectively."
        )

    # it's a normal-only plot, so there should be at least one normal image
    _validate_atleast_one_normal_image(image_classes)
    _validate_perimg_rate_curves(fprs[image_classes == 0], nan_allowed=False)

    # ** compute **

    # there may be `nan`s but only in the anomalous images
    # in the curves of normal images, there should NOT be `nan`s
    if (image_classes == 0).any():
        _validate_perimg_rate_curves(fprs[image_classes == 0], nan_allowed=False)

    fprs_norm = fprs[image_classes == 0]
    mean = fprs_norm.mean(dim=0)
    min_ = fprs_norm.min(dim=0)[0]
    max_ = fprs_norm.max(dim=0)[0]
    num_norm_images = fprs_norm.shape[0]
    sem = fprs.std(dim=0) / torch.sqrt(torch.tensor(num_norm_images))

    fig, ax = plt.subplots(figsize=(7, 7)) if ax is None else (None, ax)

    # ** plot [perimg] **
    _plot_perimg_curves(
        ax,
        shared_fpr,
        fprs,
        *[
            # don't show in the legend
            dict(label=None) if imgclass == 0 else
            # don't plot anomalous images
            None
            for imgclass in image_classes
        ],
        # shared kwargs
        linewidth=0.5,
        alpha=0.3,
    )

    # ** plot [stats] **
    ax.plot(shared_fpr, mean, color="black", linewidth=2, linestyle="--", label="mean")
    ax.plot(shared_fpr, min_, color="green", linewidth=2, linestyle="--", label="min")
    ax.plot(shared_fpr, max_, color="orange", linewidth=2, linestyle="--", label="max")
    ax.fill_between(shared_fpr, mean - 3 * sem, mean + 3 * sem, color="black", alpha=0.3, label="3 SEM (mean's 99% CI)")

    # ** format **
    _format_axis_rate_metric_linear(ax, axis=0)
    ax.set_xlabel("Shared FPR")
    _format_axis_rate_metric_linear(ax, axis=1)
    ax.set_ylabel("In-Image FPR (or Y-axis-wise Statistic)")
    ax.set_title("FPR: Shared vs In-Image Curves (Norm. Images Only)")
    ax.legend(loc="lower right", fontsize="small", title_fontsize="small", title="Stats across images")

    return fig, ax


def plot_th_fpr_curves_norm_only(
    fprs: Tensor,
    shared_fpr: Tensor,
    thresholds: Tensor,
    image_classes: Tensor,
    th_lb_fpr_ub: tuple[Tensor | float, Tensor | float] | None = None,
    th_ub_fpr_lb: tuple[Tensor | float, Tensor | float] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure | None, Axes]:
    """Plot thresholds vs FPR curves, only from normal images along with their shared fpr metric.

    Args:
        fprs: shape (num_images, num_thresholds) monotonically decreasing
        shared_fpr: shape (num_thresholds,) monotonically decreasing
        thresholds: shape (num_thresholds,) strictly increasing
        image_classes: shape (num_images,)
        th_lb_fpr_ub: (th_lower_bound, fpr_upper_bound)
        th_ub_fpr_lb: (th_upper_bound, fpr_lower_bound)
        ax: matplotlib Axes
    Returns:
        fig, ax
            fig: matplotlib Figure
            ax: matplotlib Axes
    """

    # ** validate **
    _validate_thresholds(thresholds)
    _validate_rate_curve(shared_fpr)
    _validate_perimg_rate_curves(fprs, nan_allowed=True)  # anomalous images may have `nan`s if all pixels are anomalous
    _validate_image_classes(image_classes)

    if fprs.shape[0] != image_classes.shape[0]:
        raise ValueError(
            f"Expected argument `fprs` to have the same number of images as argument `image_classes`, "
            f"but got {fprs.shape[0]} images and {image_classes.shape[0]} images, respectively."
        )

    # it's a normal-only plot, so there should be at least one normal image
    _validate_atleast_one_normal_image(image_classes)
    _validate_perimg_rate_curves(fprs[image_classes == 0], nan_allowed=False)

    def _validate_and_convert_bound_tuple(tup: tuple[Tensor | float, Tensor | float | int]) -> tuple[Tensor, Tensor]:
        if not isinstance(tup, Sequence):
            raise TypeError(f"Expected argument to be a sequence, but got {type(tup)}.")

        if len(tup) != 2:
            raise ValueError(f"Expected argument to be of size 2, but got size {len(tup)}.")

        th: Tensor | float
        fpr: Tensor | float | int
        th, fpr = tup  # type: ignore
        th = _validate_and_convert_threshold(th)
        fpr = _validate_and_convert_rate(fpr, nonzero=False, nonone=False)
        return th, fpr

    if th_lb_fpr_ub is None:
        # default lower/upper bound is the lowest threshold / 1
        th_lbound = thresholds[0]
        fpr_ubound = torch.as_tensor(1.0)
    else:
        th_lbound, fpr_ubound = _validate_and_convert_bound_tuple(th_lb_fpr_ub)

    if th_ub_fpr_lb is None:
        # default upper/lower bound is the highest threshold / 0
        th_ubound = thresholds[-1]
        fpr_lbound = torch.as_tensor(0.0)
    else:
        th_ubound, fpr_lbound = _validate_and_convert_bound_tuple(th_ub_fpr_lb)

    if th_lbound > th_ubound:
        raise ValueError(
            f"Expected threshold lower bound to be less than upper bound, but got {th_lbound} > {th_ubound}."
        )

    if fpr_lbound > fpr_ubound:
        raise ValueError(f"Expected FPR lower bound to be less than upper bound, but got {fpr_lbound} > {fpr_ubound}.")

    fig, ax = plt.subplots(figsize=(7, 7)) if ax is None else (None, ax)

    # ** plot [curves] **
    _plot_perimg_curves(
        ax,
        thresholds,
        fprs,
        *[
            # don't show in the legend
            dict(label=None) if imgclass == 0 else
            # don't plot anomalous images
            None
            for imgclass in image_classes
        ],
        # shared kwargs
        linewidth=0.5,
        alpha=0.3,
        color="gray",
    )
    ax.plot(thresholds, shared_fpr, color="black", linewidth=2, linestyle="--", label="Mean")

    # ** plot [bounds] **
    if th_lb_fpr_ub is not None:
        ax.axhline(
            fpr_ubound,
            label=f"Shared FPR upper bound ({float(100 * fpr_ubound):.2g}%)",
            linestyle="--",
            color="black",
        )
        ax.axvline(
            th_lbound,
            label="Threshold lower bound (@ FPR upper bound)",
            linestyle="-.",
            color="black",
        )

    if th_ub_fpr_lb is not None:
        ax.axhline(
            fpr_lbound,
            label=f"Shared FPR lower bound ({float(100 * fpr_lbound):.2g}%)",
            linestyle="--",
            color="gray",
        )
        ax.axvline(
            th_ubound,
            label="Threshold upper bound (@ FPR lower bound)",
            linestyle="-.",
            color="gray",
        )

    # ** plot [rectangle] **
    ax.add_patch(
        Rectangle(
            (th_lbound, fpr_lbound),
            th_ubound - th_lbound,
            fpr_ubound - fpr_lbound,
            facecolor="cyan",
            alpha=0.2,
        )
    )

    # ** format **
    ax.set_xlim(thresholds[0], thresholds[-1])
    ax.set_xlabel("Thresholds")

    _format_axis_rate_metric_linear(ax, axis=1)
    ax.set_ylabel("False Positive Rate (FPR)")

    ax.set_title("Thresholds vs FPR on Normal Images")

    ax.legend(loc="upper right", fontsize="small", title_fontsize="small")

    return fig, ax


# =========================================== COMPARE ===========================================


def compare_models_perimg(
    models: dict[str, Tensor],
    metric_name: str,
    random_model_score: float | Tensor | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot image index vs metric scatter plot for each model.

    The average of the metric is also plotted as a horizontal line.

    Args:
        models: dict of (model_name, metric_tensor) pairs
        metric_name: name of the metric
        random_model_score: score of the random model (for reference; ex: 0.5 for AUROC)
                            if given, a horizontal line will be added at this score
        ax: matplotlib Axes

    Returns:
        fig, ax
    """
    MARKERS = [
        "o",
        "X",
        "s",
        "P",
        "*",
        "D",
        "v",
        "^",
        "<",
        ">",
        "p",
    ]

    # ** validate **
    models = _validate_and_convert_models_dict(models)

    # ** plot **

    num_imgs = next(iter(models.values())).shape[0]
    if ax is None:
        # get the number of images from an arbitrary model (they all have the same number of images)
        fig, ax = plt.subplots(figsize=(min(num_imgs / 7, 20), 7))
    else:
        fig, ax = None, ax

    df = pd.DataFrame(models).reset_index()  # column `index` is the image index
    df = df.melt(id_vars=["index"], value_vars=list(models.keys()), var_name="model", value_name="metric")
    df.sort_values("model", inplace=True)

    for modelidx, (model, df_model) in enumerate(df.groupby("model")):
        avg = df_model["metric"].mean()
        scatter = ax.scatter(
            df_model["index"],
            df_model["metric"],
            label=f"{model} (avg={avg:.2%})",
            marker=MARKERS[modelidx % len(MARKERS)],
            alpha=0.8,
        )
        ax.axhline(avg, color=scatter.get_facecolor(), linestyle="--")

    # ** format **

    if random_model_score is not None:
        _add_avline_at_score_random_model(ax, random_model_score, axis=0, add_legend=False)

    ax.legend(loc="lower right", title="Model")

    # Y-axis
    _format_axis_rate_metric_linear(ax, axis=1)
    _format_axis_rate_metric_linear(ax.twinx(), axis=1)  # for the right axis
    ax.grid(axis="y", which="major")
    ax.grid(axis="y", which="minor", linestyle="--", alpha=0.5)
    ax.set_ylabel(f"{metric_name} [%]")

    # X-axis
    _format_axis_imgidx(ax, num_imgs)

    ax.set_title(f"Image Index vs. {metric_name}")

    return fig, ax


def compare_models_perimg_rank(
    models: dict[str, Tensor],
    metric_name: str,
    higher_is_better: bool = True,
    atol: float | None = 0.001,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot image index vs rank scatter plot for each model.

    Rank is from 1 to `num_models`, where 1 is the best model.
    Average rank is also plotted as a horizontal line.
    Red lines are drawn between adjacent ranks whose metric values are within `atol` of each other.

    Args:
        models: dict of (model_name, metric_tensor) pairs
        metric_name: name of the metric
        higher_is_better: whether a higher metric value means "better model"
        atol: absolute tolerance for the metric values to be considered "equal"
        ax: matplotlib Axes
    Returns:
        fig, ax
    """

    MARKERS = [
        "o",
        "X",
        "s",
        "P",
        "*",
        "D",
        "v",
        "^",
        "<",
        ">",
        "p",
    ]

    # ** validate **
    models = _validate_and_convert_models_dict(models)

    if atol is not None:
        atol = float(_validate_and_convert_rate(atol, nonzero=True, nonone=False))

    # ** plot **

    # get the number of images from an arbitrary model (they all have the same number of images)
    num_imgs = next(iter(models.values())).shape[0]
    num_models = len(models)

    if ax is None:
        fig, ax = plt.subplots(figsize=(min(num_imgs / 7, 20), min(10, 2 * max(3, num_models))))
    else:
        fig, ax = None, ax

    def get_rank(row, higher_is_better):
        rank = scipy.stats.rankdata(-row.values if higher_is_better else row.values, method="average")
        return dict(zip(row.index, rank))

    df = pd.DataFrame(models)
    df.dropna(inplace=True, how="any", axis=0)
    df = df.apply(get_rank, higher_is_better=higher_is_better, axis=1, result_type="expand")
    df = df.reset_index()  # column `index` is the image index
    df = df.melt(id_vars=["index"], value_vars=list(models.keys()), var_name="model", value_name="rank")
    df.sort_values("model", inplace=True)

    for modelidx, (model, df_model) in enumerate(df.groupby("model")):
        avg = df_model["rank"].mean()
        scatter = ax.scatter(
            df_model["index"],
            df_model["rank"],
            label=f"{model} (avg={avg:.2f})",
            marker=MARKERS[modelidx % len(MARKERS)],
            zorder=2.1,  # above lines, including grid, default is `2`
        )
        ax.axhline(avg, color=scatter.get_facecolor(), linestyle="--")

    if atol is not None:
        # add a red line between adjacent ranks that are within `atol` of each other

        # get all comparisons of all models (not against oneself), and the difference in metric
        # `within_tolerance[imgidx, model1, model2]` = True if within tolerance, False otherwise
        # where model1 and model2 are keys in `models`
        # `tmp` is just to make the code below easier to read
        tmp = pd.DataFrame(models)
        tmp.dropna(inplace=True, how="any", axis=0)
        tmp = tmp.reset_index()  # column `index` is the image index
        tmp = tmp.melt(id_vars=["index"], value_vars=list(models.keys()), var_name="model", value_name="metric")
        tmp = tmp.merge(tmp, on="index", suffixes=("_1", "_2"))
        tmp["difference"] = (tmp["metric_1"] - tmp["metric_2"]).abs()
        tmp["within_tolerance"] = tmp["difference"] <= atol
        tmp = tmp.set_index(["index", "model_1", "model_2"])
        within_tolerance = tmp

        # map (imgidx, rank) -> model name
        imgidx_rank_2_model = df.set_index(["index", "rank"]).sort_index()

        # # at each image, consider all rank comparisons: (1, 2), (1, 3), ..., (2, 3), ..., (num_models - 1, num_models)
        # rank_comparisons = list(itertools.combinations(range(1, num_models + 1), 2))

        # excluding the `nan`s
        imgidxs = df["index"].unique()

        atleast_one_tie = False
        for imgidx in imgidxs:
            rank_2_model = imgidx_rank_2_model.loc[imgidx]
            # it can be (1, 2, 3), but also (1.5, 1.5, 3) for example
            ranks = rank_2_model.index.unique()
            for rank1, rank2 in itertools.combinations(ranks, 2):
                model1 = rank_2_model.loc[rank1, "model"]
                model2 = rank_2_model.loc[rank2, "model"]
                # get an arbitrary choice in case of ties
                model1 = model1.iloc[0] if not isinstance(model1, str) else model1
                model2 = model2.iloc[0] if not isinstance(model2, str) else model2
                is_within = within_tolerance.loc[(imgidx, model1, model2)]["within_tolerance"]
                if is_within:
                    ax.plot(
                        [imgidx, imgidx],
                        [rank1, rank2],
                        color="red",
                        linewidth=2,
                        label=f"Within Tolerance ({atol:.2%})" if not atleast_one_tie else None,
                    )
                    atleast_one_tie = True

    # ** format **

    add_legend_entry_for_ties = (atol is not None) and atleast_one_tie
    ax.legend(
        loc="upper right",
        title="Model",
        ncol=len(models) + add_legend_entry_for_ties,
        fontsize="small",
        title_fontsize="small",
    )

    # Y-axis
    ax.yaxis.set_major_locator(IndexLocator(1, 0))
    ax.set_ylim(0, len(models) + 1)
    ax.set_ylabel("Rank (lower is better)")
    ax.invert_yaxis()

    # X-axis
    _format_axis_imgidx(ax, num_imgs)

    ax.set_title(f"Image Index vs. {metric_name} Rank")

    return fig, ax
