"""Helper functions to generate ROC-style plots of various metrics."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from matplotlib import pyplot as plt
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from torch import Tensor


def plot_figure(
    x_vals: Tensor,
    y_vals: Tensor,
    auc: Tensor,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlabel: str,
    ylabel: str,
    loc: str,
    title: str,
    sample_points: int = 1000,
) -> tuple[Figure, Axis]:
    """Generate a simple, ROC-style plot, where x_vals is plotted against y_vals.

    Note that a subsampling is applied if > sample_points are present in x/y, as matplotlib plotting draws
    every single plot which takes very long, especially for high-resolution segmentations.

    Args:
        x_vals (Tensor): x values to plot
        y_vals (Tensor): y values to plot
        auc (Tensor): normalized area under the curve spanned by x_vals, y_vals
        xlim (tuple[float, float]): displayed range for x-axis
        ylim (tuple[float, float]): displayed range for y-axis
        xlabel (str): label of x axis
        ylabel (str): label of y axis
        loc (str): string-based legend location, for details see
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
        title (str): title of the plot
        sample_points (int): number of sampling points to subsample x_vals/y_vals with

    Returns:
        tuple[Figure, Axis]: Figure and the contained Axis
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
