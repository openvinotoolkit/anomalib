"""  TODO
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy
import torch
from matplotlib.axes import Axes
from matplotlib.pyplot import Figure
from matplotlib.ticker import FixedLocator, LogFormatter, PercentFormatter
from torch import Tensor

from .common import (
    _perimg_boxplot_stats,
    _validate_atleast_one_anomalous_image,
    _validate_aucs,
    _validate_image_class,
    _validate_image_classes,
    _validate_perimg_rate_curves,
    _validate_rate_curve,
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


def _format_axis_rate_metric_log(ax: Axes, axis: int, lower_lim: float = 1e-3, num_ticks_major: int = 6) -> None:
    """will be used in a later PR
    TODO remove this comment
    """

    if not isinstance(ax, Axes) or not isinstance(axis, int):
        raise ValueError("Expected arguments `ax` to be an Axes and `axis` to be an integer.")

    lims = (lower_lim, 1)
    lower_lim_rounded_exponent = int(numpy.floor(numpy.log10(lower_lim)))

    ticks_major = numpy.logspace(lower_lim_rounded_exponent, 0, abs(lower_lim_rounded_exponent) + 1)
    formatter_major = LogFormatter()
    ticks_minor = numpy.logspace(lower_lim_rounded_exponent, 0, 2 * abs(lower_lim_rounded_exponent) + 1)

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

    if only_class is None and only_class not in image_classes:
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
    _validate_aucs(aucs, nan_allowed=True)
    _validate_atleast_one_anomalous_image(image_classes)

    fig, ax = plt.subplots() if ax is None else (None, ax)

    bp_stats = _perimg_boxplot_stats(aucs, image_classes, only_class=1)

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
            (dict(label=f"idx={imgidx:03}") if img_cls == 1 else None)  # a generic label  # don't plot this curve
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
    ax.set_title("Per-Image Overlap Curves")

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
                  refer to `anomalib.utils.metrics.perimg.common._perimg_boxplot_stats()`

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
    ax.set_title("Per-Image Overlap Curves (AUC boxplot statistics)")

    return fig, ax
