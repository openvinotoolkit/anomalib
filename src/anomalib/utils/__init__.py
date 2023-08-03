"""Helpers for downloading files, calculating metrics, computing anomaly maps, and visualization."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy
from line_profiler import LineProfiler


def plot_cumulative_histogram(
    good_scores: numpy.ndarray, defect_scores: numpy.ndarray, threshold: float, save_path: Optional[str]
) -> None:
    """Plot a cumulative histogram of scores.

    Args:
        good_scores: scores of good samples
        defect_scores: scores of defect samples
        threshold: threshold to separate good and defect samples
        save_path: path to save the plot, if None the plot will be showed

    Returns: None
    """
    _, axes = plt.subplots(figsize=(8, 8))
    axes.set(yticklabels=[])
    axes.tick_params(left=False)

    if len(defect_scores) > 0:
        axes.hist(
            defect_scores,
            bins=len(defect_scores),
            density=True,
            histtype="stepfilled",
            cumulative=True,
            facecolor=(1, 0, 0, 0.5),
        )

    if len(good_scores) > 0:
        axes.hist(
            good_scores,
            bins=len(good_scores),
            density=True,
            histtype="stepfilled",
            cumulative=-1,
            facecolor=(0, 1, 0, 0.5),
        )

    axes.axvline(x=threshold, color="r", linestyle="--")

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "cumulative_histogram.png"))
    else:
        plt.show()


def profile(func):
    """Decorator to profile a function with line profile."""
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper function."""
        prof = LineProfiler()
        try:
            return prof(func)(*args, **kwargs)
        finally:
            prof.print_stats()

    return wrapper
