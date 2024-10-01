"""Tests for per-image binary classification curves using numpy version."""

# Original Code
# https://github.com/jpcbertoldo/aupimo
#
# Modified
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: SLF001, PT011

import pytest
import torch

from anomalib.metrics.pimo.binary_classification_curve import (
    _binary_classification_curve,
    binary_classification_curve,
    per_image_fpr,
    per_image_tpr,
    threshold_and_binary_classification_curve,
)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Generate test cases."""
    pred = torch.arange(1, 5, dtype=torch.float32)
    thresholds = torch.arange(1, 5, dtype=torch.float32)

    gt_norm = torch.zeros(4).to(bool)
    gt_anom = torch.concatenate([torch.zeros(2), torch.ones(2)]).to(bool)

    # in the case where thresholds are all unique values in the predictions
    expected_norm = torch.stack(
        [
            torch.tensor([[0, 4], [0, 0]]),
            torch.tensor([[1, 3], [0, 0]]),
            torch.tensor([[2, 2], [0, 0]]),
            torch.tensor([[3, 1], [0, 0]]),
        ],
        axis=0,
    ).to(int)
    expected_anom = torch.stack(
        [
            torch.tensor([[0, 2], [0, 2]]),
            torch.tensor([[1, 1], [0, 2]]),
            torch.tensor([[2, 0], [0, 2]]),
            torch.tensor([[2, 0], [1, 1]]),
        ],
        axis=0,
    ).to(int)

    expected_tprs_norm = torch.tensor([torch.nan, torch.nan, torch.nan, torch.nan])
    expected_tprs_anom = torch.tensor([1.0, 1.0, 1.0, 0.5])
    expected_tprs = torch.stack([expected_tprs_anom, expected_tprs_norm], axis=0).to(torch.float64)

    expected_fprs_norm = torch.tensor([1.0, 0.75, 0.5, 0.25])
    expected_fprs_anom = torch.tensor([1.0, 0.5, 0.0, 0.0])
    expected_fprs = torch.stack([expected_fprs_anom, expected_fprs_norm], axis=0).to(torch.float64)

    # in the case where all thresholds are higher than the highest prediction
    expected_norm_thresholds_too_high = torch.stack(
        [
            torch.tensor([[4, 0], [0, 0]]),
            torch.tensor([[4, 0], [0, 0]]),
            torch.tensor([[4, 0], [0, 0]]),
            torch.tensor([[4, 0], [0, 0]]),
        ],
        axis=0,
    ).to(int)
    expected_anom_thresholds_too_high = torch.stack(
        [
            torch.tensor([[2, 0], [2, 0]]),
            torch.tensor([[2, 0], [2, 0]]),
            torch.tensor([[2, 0], [2, 0]]),
            torch.tensor([[2, 0], [2, 0]]),
        ],
        axis=0,
    ).to(int)

    # in the case where all thresholds are lower than the lowest prediction
    expected_norm_thresholds_too_low = torch.stack(
        [
            torch.tensor([[0, 4], [0, 0]]),
            torch.tensor([[0, 4], [0, 0]]),
            torch.tensor([[0, 4], [0, 0]]),
            torch.tensor([[0, 4], [0, 0]]),
        ],
        axis=0,
    ).to(int)
    expected_anom_thresholds_too_low = torch.stack(
        [
            torch.tensor([[0, 2], [0, 2]]),
            torch.tensor([[0, 2], [0, 2]]),
            torch.tensor([[0, 2], [0, 2]]),
            torch.tensor([[0, 2], [0, 2]]),
        ],
        axis=0,
    ).to(int)

    if metafunc.function is test__binclf_one_curve:
        metafunc.parametrize(
            argnames=("pred", "gt", "thresholds", "expected"),
            argvalues=[
                (pred, gt_anom, thresholds[:3], expected_anom[:3]),
                (pred, gt_anom, thresholds, expected_anom),
                (pred, gt_norm, thresholds, expected_norm),
                (pred, gt_norm, 10 * thresholds, expected_norm_thresholds_too_high),
                (pred, gt_anom, 10 * thresholds, expected_anom_thresholds_too_high),
                (pred, gt_norm, 0.001 * thresholds, expected_norm_thresholds_too_low),
                (pred, gt_anom, 0.001 * thresholds, expected_anom_thresholds_too_low),
            ],
        )

    preds = torch.stack([pred, pred], axis=0)
    gts = torch.stack([gt_anom, gt_norm], axis=0)
    binclf_curves = torch.stack([expected_anom, expected_norm], axis=0)
    binclf_curves_thresholds_too_high = torch.stack(
        [expected_anom_thresholds_too_high, expected_norm_thresholds_too_high],
        axis=0,
    )
    binclf_curves_thresholds_too_low = torch.stack(
        [expected_anom_thresholds_too_low, expected_norm_thresholds_too_low],
        axis=0,
    )

    if metafunc.function is test__binclf_multiple_curves:
        metafunc.parametrize(
            argnames=("preds", "gts", "thresholds", "expecteds"),
            argvalues=[
                (preds, gts, thresholds[:3], binclf_curves[:, :3]),
                (preds, gts, thresholds, binclf_curves),
            ],
        )

    if metafunc.function is test_binclf_multiple_curves:
        metafunc.parametrize(
            argnames=(
                "preds",
                "gts",
                "thresholds",
                "expected_binclf_curves",
            ),
            argvalues=[
                (preds[:1], gts[:1], thresholds, binclf_curves[:1]),
                (preds, gts, thresholds, binclf_curves),
                (10 * preds, gts, 10 * thresholds, binclf_curves),
            ],
        )

    if metafunc.function is test_binclf_multiple_curves_validations:
        metafunc.parametrize(
            argnames=("args", "kwargs", "exception"),
            argvalues=[
                # `scores` and `gts` must be 2D
                ([preds.reshape(2, 2, 2), gts, thresholds], {}, ValueError),
                ([preds, gts.flatten(), thresholds], {}, ValueError),
                # `thresholds` must be 1D
                ([preds, gts, thresholds.reshape(2, 2)], {}, ValueError),
                # `scores` and `gts` must have the same shape
                ([preds, gts[:1], thresholds], {}, ValueError),
                ([preds[:, :2], gts, thresholds], {}, ValueError),
                # `scores` be of type float
                ([preds.to(int), gts, thresholds], {}, TypeError),
                # `gts` be of type bool
                ([preds, gts.to(int), thresholds], {}, TypeError),
                # `thresholds` be of type float
                ([preds, gts, thresholds.to(int)], {}, TypeError),
                # `thresholds` must be sorted in ascending order
                ([preds, gts, torch.flip(thresholds, dims=[0])], {}, ValueError),
                ([preds, gts, torch.concatenate([thresholds[-2:], thresholds[:2]])], {}, ValueError),
                # `thresholds` must be unique
                ([preds, gts, torch.sort(torch.concatenate([thresholds, thresholds]))[0]], {}, ValueError),
            ],
        )

    # the following tests are for `per_image_binclf_curve()`, which expects
    # inputs in image spatial format, i.e. (height, width)
    preds = preds.reshape(2, 2, 2)
    gts = gts.reshape(2, 2, 2)

    per_image_binclf_curves_argvalues = [
        # `thresholds_choice` = "given"
        (
            preds,
            gts,
            "given",
            thresholds,
            None,
            thresholds,
            binclf_curves,
        ),
        (
            preds,
            gts,
            "given",
            10 * thresholds,
            2,
            10 * thresholds,
            binclf_curves_thresholds_too_high,
        ),
        (
            preds,
            gts,
            "given",
            0.01 * thresholds,
            None,
            0.01 * thresholds,
            binclf_curves_thresholds_too_low,
        ),
        # `thresholds_choice` = 'minmax-linspace'"
        (
            preds,
            gts,
            "minmax-linspace",
            None,
            len(thresholds),
            thresholds,
            binclf_curves,
        ),
        (
            2 * preds,
            gts.to(int),  # this is ok
            "minmax-linspace",
            None,
            len(thresholds),
            2 * thresholds,
            binclf_curves,
        ),
    ]

    if metafunc.function is test_per_image_binclf_curve:
        metafunc.parametrize(
            argnames=(
                "anomaly_maps",
                "masks",
                "threshold_choice",
                "thresholds",
                "num_thresholds",
                "expected_thresholds",
                "expected_binclf_curves",
            ),
            argvalues=per_image_binclf_curves_argvalues,
        )

    if metafunc.function is test_per_image_binclf_curve_validations:
        metafunc.parametrize(
            argnames=("args", "exception"),
            argvalues=[
                # `scores` and `gts` must be 3D
                ([preds.reshape(2, 2, 2, 1), gts], ValueError),
                ([preds, gts.flatten()], ValueError),
                # `scores` and `gts` must have the same shape
                ([preds, gts[:1]], ValueError),
                ([preds[:, :1], gts], ValueError),
                # `scores` be of type float
                ([preds.to(int), gts], TypeError),
                # `gts` be of type bool or int
                ([preds, gts.to(float)], TypeError),
                # `thresholds` be of type float
                ([preds, gts, thresholds.to(int)], TypeError),
            ],
        )
        metafunc.parametrize(
            argnames=("kwargs",),
            argvalues=[
                (
                    {
                        "threshold_choice": "minmax-linspace",
                        "thresholds": None,
                        "num_thresholds": len(thresholds),
                    },
                ),
            ],
        )

    # same as above but testing other validations
    if metafunc.function is test_per_image_binclf_curve_validations_alt:
        metafunc.parametrize(
            argnames=("args", "kwargs", "exception"),
            argvalues=[
                # invalid `thresholds_choice`
                (
                    [preds, gts],
                    {"threshold_choice": "glfrb", "thresholds": thresholds, "num_thresholds": None},
                    ValueError,
                ),
            ],
        )

    if metafunc.function is test_rate_metrics:
        metafunc.parametrize(
            argnames=("binclf_curves", "expected_fprs", "expected_tprs"),
            argvalues=[
                (binclf_curves, expected_fprs, expected_tprs),
                (10 * binclf_curves, expected_fprs, expected_tprs),
            ],
        )


# ==================================================================================================
# LOW-LEVEL FUNCTIONS (PYTHON)


def test__binclf_one_curve(
    pred: torch.Tensor,
    gt: torch.Tensor,
    thresholds: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    """Test if `_binclf_one_curve()` returns the expected values."""
    computed = _binary_classification_curve(pred, gt, thresholds)
    assert computed.shape == (thresholds.numel(), 2, 2)
    assert (computed == expected.numpy()).all()


def test__binclf_multiple_curves(
    preds: torch.Tensor,
    gts: torch.Tensor,
    thresholds: torch.Tensor,
    expecteds: torch.Tensor,
) -> None:
    """Test if `_binclf_multiple_curves()` returns the expected values."""
    computed = binary_classification_curve(preds, gts, thresholds)
    assert computed.shape == (preds.shape[0], thresholds.numel(), 2, 2)
    assert (computed == expecteds).all()


# ==================================================================================================
# API FUNCTIONS (NUMPY)


def test_binclf_multiple_curves(
    preds: torch.Tensor,
    gts: torch.Tensor,
    thresholds: torch.Tensor,
    expected_binclf_curves: torch.Tensor,
) -> None:
    """Test if `binclf_multiple_curves()` returns the expected values."""
    computed = binary_classification_curve(
        preds,
        gts,
        thresholds,
    )
    assert computed.shape == expected_binclf_curves.shape
    assert (computed == expected_binclf_curves).all()

    # it's ok to have the threhsholds beyond the range of the preds
    binary_classification_curve(preds, gts, 2 * thresholds)

    # or inside the bounds without reaching them
    binary_classification_curve(preds, gts, 0.5 * thresholds)

    # it's also ok to have more thresholds than unique values in the preds
    # add the values in between the thresholds
    thresholds_unncessary = 0.5 * (thresholds[:-1] + thresholds[1:])
    thresholds_unncessary = torch.concatenate([thresholds_unncessary, thresholds])
    thresholds_unncessary = torch.sort(thresholds_unncessary)[0]
    binary_classification_curve(preds, gts, thresholds_unncessary)

    # or less
    binary_classification_curve(preds, gts, thresholds[1:3])


def test_binclf_multiple_curves_validations(args: list, kwargs: dict, exception: Exception) -> None:
    """Test if `_binclf_multiple_curves_python()` raises the expected errors."""
    with pytest.raises(exception):
        binary_classification_curve(*args, **kwargs)


def test_per_image_binclf_curve(
    anomaly_maps: torch.Tensor,
    masks: torch.Tensor,
    threshold_choice: str,
    thresholds: torch.Tensor | None,
    num_thresholds: int | None,
    expected_thresholds: torch.Tensor,
    expected_binclf_curves: torch.Tensor,
) -> None:
    """Test if `per_image_binclf_curve()` returns the expected values."""
    computed_thresholds, computed_binclf_curves = threshold_and_binary_classification_curve(
        anomaly_maps,
        masks,
        threshold_choice=threshold_choice,
        thresholds=thresholds,
        num_thresholds=num_thresholds,
    )

    # thresholds
    assert computed_thresholds.shape == expected_thresholds.shape
    assert computed_thresholds.dtype == computed_thresholds.dtype
    assert (computed_thresholds == expected_thresholds).all()

    # binclf_curves
    assert computed_binclf_curves.shape == expected_binclf_curves.shape
    assert computed_binclf_curves.dtype == expected_binclf_curves.dtype
    assert (computed_binclf_curves == expected_binclf_curves).all()


def test_per_image_binclf_curve_validations(args: list, kwargs: dict, exception: Exception) -> None:
    """Test if `per_image_binclf_curve()` raises the expected errors."""
    with pytest.raises(exception):
        threshold_and_binary_classification_curve(*args, **kwargs)


def test_per_image_binclf_curve_validations_alt(args: list, kwargs: dict, exception: Exception) -> None:
    """Test if `per_image_binclf_curve()` raises the expected errors."""
    test_per_image_binclf_curve_validations(args, kwargs, exception)


def test_rate_metrics(
    binclf_curves: torch.Tensor,
    expected_fprs: torch.Tensor,
    expected_tprs: torch.Tensor,
) -> None:
    """Test if rate metrics are computed correctly."""
    tprs = per_image_tpr(binclf_curves)
    fprs = per_image_fpr(binclf_curves)

    assert tprs.shape == expected_tprs.shape
    assert fprs.shape == expected_fprs.shape

    assert torch.allclose(tprs, expected_tprs, equal_nan=True)
    assert torch.allclose(fprs, expected_fprs, equal_nan=True)
