"""Helper functions to generate ROC-style plots of various metrics.

This module provides utility functions for generating ROC-style plots and other
visualization helpers used by metrics in Anomalib.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from matplotlib import pyplot as plt
from matplotlib.axis import Axis
from matplotlib.figure import Figure


def plot_figure(
    x_vals: torch.Tensor,
    y_vals: torch.Tensor,
    auc: torch.Tensor,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlabel: str,
    ylabel: str,
    loc: str,
    title: str,
    sample_points: int = 1000,
) -> tuple[Figure, Axis]:
    """Generate a ROC-style plot with x values plotted against y values.

    The function creates a matplotlib figure with a single axis showing the curve
    defined by ``x_vals`` and ``y_vals``. If the number of points exceeds
    ``sample_points``, the data is subsampled to improve plotting performance.

    Args:
        x_vals (torch.Tensor): Values to plot on x-axis.
        y_vals (torch.Tensor): Values to plot on y-axis.
        auc (torch.Tensor): Area under curve value to display in legend.
        xlim (tuple[float, float]): Display range for x-axis as ``(min, max)``.
        ylim (tuple[float, float]): Display range for y-axis as ``(min, max)``.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        loc (str): Legend location. See matplotlib documentation for valid values:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
        title (str): Title of the plot.
        sample_points (int, optional): Maximum number of points to plot. Data will
            be subsampled if it exceeds this value. Defaults to ``1000``.

    Returns:
        tuple[Figure, Axis]: Tuple containing the figure and its main axis.

    Example:
        >>> import torch
        >>> x = torch.linspace(0, 1, 100)
        >>> y = x ** 2
        >>> auc = torch.tensor(0.5)
        >>> fig, ax = plot_figure(
        ...     x_vals=x,
        ...     y_vals=y,
        ...     auc=auc,
        ...     xlim=(0, 1),
        ...     ylim=(0, 1),
        ...     xlabel="False Positive Rate",
        ...     ylabel="True Positive Rate",
        ...     loc="lower right",
        ...     title="ROC Curve",
        ... )
    """
    fig, axis = plt.subplots()

    x_vals = x_vals.detach().cpu()
    y_vals = y_vals.detach().cpu()

    if sample_points < x_vals.size(0):
        possible_idx = range(x_vals.size(0))
        interval = len(possible_idx) // sample_points

        idx = [0]  # make sure to start at first point
        idx.extend(possible_idx[::interval])
        idx.append(possible_idx[-1])  # also include last point

        idx = torch.tensor(
            idx,
            device=x_vals.device,
        )
        x_vals = torch.index_select(x_vals, 0, idx)
        y_vals = torch.index_select(y_vals, 0, idx)

    axis.plot(
        x_vals,
        y_vals,
        color="darkorange",
        figure=fig,
        lw=2,
        label=f"AUC: {auc.detach().cpu():0.2f}",
    )

    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend(loc=loc)
    axis.set_title(title)
    return fig, axis
