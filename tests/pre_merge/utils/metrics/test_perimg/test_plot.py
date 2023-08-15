"""Tests for plotting functions in the per-image metrics module."""

import matplotlib.pyplot as plt
import torch

from anomalib.utils.metrics.perimg.plot import (
    _plot_perimg_curves,
    plot_all_pimo_curves,
)


def pytest_generate_tests(metafunc):
    # curves with same general, monotonic shape as tprs/fprs curves
    ys = (torch.linspace(-10, 10, 300)[None, :] + torch.arange(1, 5)[:, None]).sigmoid().flip(1)
    th = torch.linspace(0, 15, 300)
    cases = [
        (th, ys),  # th vs tpr like
        (ys[0], ys),  # fpr vs tpr like
    ]

    if metafunc.function is test__plot_perimg_curves:
        metafunc.parametrize(
            argnames=("x", "ys"),
            argvalues=cases,
        )

    if metafunc.function is test__plot_perimg_curves_kwargs:
        metafunc.parametrize(
            argnames=("x", "ys"),
            argvalues=cases[1:2],
        )


def test__plot_perimg_curves(x, ys):
    """Test _plot_perimg_curves."""
    fig, ax = plt.subplots()
    _plot_perimg_curves(ax, x, ys, *[{} for _ in range(ys.shape[0])])


def test__plot_perimg_curves_kwargs(x, ys):
    fig, ax = plt.subplots()
    _plot_perimg_curves(
        ax,
        x,
        ys,
        # per-curve kwargs
        *[dict(linewidth=i) for i in range(ys.shape[0])],
        # shared kwargs
        color="blue",
    )


def test_plot_all_pimo_curves():
    """Test plot_all_pimo_curves."""
    rates = (torch.linspace(-10, 10, 300)[None, :] + torch.arange(1, 5)[:, None]).sigmoid().flip(1)
    img_cls = (torch.arange(1, 5) % 2).to(torch.int32)
    fig, ax = plot_all_pimo_curves(rates[0], rates, img_cls)
    plot_all_pimo_curves(rates[0], rates, img_cls, ax=ax)
