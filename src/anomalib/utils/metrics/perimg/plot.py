"""  TODO
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy
from matplotlib.axes import Axes
from matplotlib.pyplot import Figure
from matplotlib.ticker import FixedLocator, LogFormatter, PercentFormatter
from torch import Tensor

from .common import (
    _validate_atleast_one_anomalous_image,
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


def _format_axis_rate_metric_log(ax: Axes, axis: int, lower_lim: float = 1e-3) -> None:
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


# =========================================== GENERIC ===========================================


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
    ax.set_title("Per-Image Overlap Curves")

    return fig, ax
