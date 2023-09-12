"""Tests for plotting functions in the per-image metrics module."""

import matplotlib.pyplot as plt
import torch


from anomalib.utils.metrics.perimg.common import _perimg_boxplot_stats
from anomalib.utils.metrics.perimg.plot import (
    _plot_perimg_curves,
    _plot_perimg_metric_boxplot,
    plot_all_pimo_curves,
    plot_aupimo_boxplot,
    plot_boxplot_pimo_curves,
    plot_pimfpr_curves_norm_only,
    plot_pimfpr_curves_norm_vs_anom,
    plot_th_fpr_curves_norm_only,
)


def pytest_generate_tests(metafunc):
    # curves with same general, monotonic shape as tprs/fprs curves
    rates = (torch.linspace(-10, 10, 300)[None, :] + torch.arange(1, 15)[:, None]).sigmoid().flip(1)
    th = torch.linspace(0, 15, 300)
    img_cls = (torch.arange(1, 15) % 3 == 0).to(torch.int32)
    aucs = rates.mean(1)

    if metafunc.function is test__plot_perimg_curves:
        metafunc.parametrize(
            argnames=("x", "ys"),
            argvalues=[
                (th, rates),  # th vs tpr like
                (rates[0], rates),  # fpr vs tpr like
            ],
        )

    if metafunc.function is test__plot_perimg_curves_kwargs:
        metafunc.parametrize(
            argnames=("x", "ys"),
            argvalues=[
                (rates[0], rates),
            ],
        )

    if "rates" in metafunc.fixturenames:
        metafunc.parametrize(("rates",), [(rates,)])

    if "image_classes" in metafunc.fixturenames:
        metafunc.parametrize(("image_classes",), [(img_cls,)])

    if "aucs" in metafunc.fixturenames:
        metafunc.parametrize(("aucs",), [(aucs,)])

    if "only_class" in metafunc.fixturenames:
        metafunc.parametrize(
            ("only_class",),
            [
                (None,),
                (0,),
                (1,),
            ],
        )

    if "thresholds" in metafunc.fixturenames:
        metafunc.parametrize(
            ("thresholds",),
            [
                (th,),
            ],
        )


def test__plot_perimg_curves(x, ys):
    """Test _plot_perimg_curves."""
    _, ax = plt.subplots()
    _plot_perimg_curves(ax, x, ys, *[{} for _ in range(ys.shape[0])])


def test__plot_perimg_curves_kwargs(x, ys):
    _, ax = plt.subplots()
    _plot_perimg_curves(
        ax,
        ys[0],
        ys,
        # per-curve kwargs
        *[dict(linewidth=i) for i in range(ys.shape[0])],
        # shared kwargs
        color="blue",
    )


def test_plot_all_pimo_curves(rates, image_classes):
    fig, ax = plot_all_pimo_curves(rates[0], rates, image_classes)
    assert fig is not None
    assert ax is not None
    plot_all_pimo_curves(rates[0], rates, image_classes, ax=ax)


def test__plot_perimg_metric_boxplot(aucs, image_classes, only_class):
    bp_stats = _perimg_boxplot_stats(aucs, image_classes)
    _, ax = plt.subplots()
    _plot_perimg_metric_boxplot(ax, aucs, image_classes, bp_stats, only_class=only_class)


def test_plot_aupimo_boxplot(aucs, image_classes):
    fig, ax = plot_aupimo_boxplot(aucs, image_classes)
    assert fig is not None
    assert ax is not None
    plot_aupimo_boxplot(aucs, image_classes, ax=ax)


def test_plot_boxplot_pimo_curves(aucs, rates, image_classes):
    bp_stats = _perimg_boxplot_stats(aucs, image_classes)
    fig, ax = plot_boxplot_pimo_curves(rates[0], rates, image_classes, bp_stats)
    assert fig is not None
    assert ax is not None
    plot_boxplot_pimo_curves(rates[0], rates, image_classes, bp_stats, ax=ax)


def test_plot_pimfpr_curves_norm_vs_anom(rates, image_classes):
    fig, ax = plot_pimfpr_curves_norm_vs_anom(rates, rates[0], image_classes)
    assert fig is not None
    assert ax is not None
    plot_pimfpr_curves_norm_vs_anom(rates, rates[0], image_classes, ax=ax)


def test_plot_pimfpr_curves_norm_only(rates, image_classes):
    fig, ax = plot_pimfpr_curves_norm_only(rates, rates[0], image_classes)
    assert fig is not None
    assert ax is not None
    plot_pimfpr_curves_norm_only(rates, rates[0], image_classes, ax=ax)


def test_plot_th_fpr_curves_norm_only(rates, thresholds, image_classes):
    bound_idx = len(thresholds) // 2
    fig, ax = plot_th_fpr_curves_norm_only(
        rates, rates[0], thresholds, image_classes, th_lbound=thresholds[bound_idx], fpr_ubound=rates[0, bound_idx]
    )
    assert fig is not None
    assert ax is not None
    plot_th_fpr_curves_norm_only(
        rates,
        rates[0],
        thresholds,
        image_classes,
        th_lbound=thresholds[bound_idx],
        fpr_ubound=rates[0, bound_idx],
        ax=ax,
    )
