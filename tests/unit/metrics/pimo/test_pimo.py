"""Test `anomalib.metrics.per_image.functional`."""

# Original Code
# https://github.com/jpcbertoldo/aupimo
#
# Modified
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest
import torch
from torch import Tensor

from anomalib.metrics.pimo import AUPIMOResult, PIMOResult, functional, pimo


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Generate tests for all functions in this module.

    All functions are parametrized with the same setting: 1 normal and 2 anomalous images.
    The anomaly maps are the same for all functions, but the masks are different.
    """
    expected_thresholds = torch.arange(1, 7 + 1, dtype=torch.float32)
    shape = (1000, 1000)  # (H, W), 1 million pixels

    # --- normal ---
    # histogram of scores:
    # value:   7   6    5    4    3    2     1
    # count:   1   9   90  900   9k   90k  900k
    # cumsum:  1  10  100   1k  10k  100k    1M
    pred_norm = torch.ones(1_000_000, dtype=torch.float32)
    pred_norm[:100_000] += 1
    pred_norm[:10_000] += 1
    pred_norm[:1_000] += 1
    pred_norm[:100] += 1
    pred_norm[:10] += 1
    pred_norm[:1] += 1
    pred_norm = pred_norm.reshape(shape)
    mask_norm = torch.zeros_like(pred_norm, dtype=torch.int32)

    expected_fpr_norm = torch.tensor([1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6], dtype=torch.float64)
    expected_tpr_norm = torch.full((7,), torch.nan, dtype=torch.float64)

    # --- anomalous ---
    pred_anom1 = pred_norm.clone()
    mask_anom1 = torch.ones_like(pred_anom1, dtype=torch.int32)
    expected_tpr_anom1 = expected_fpr_norm.clone()

    # only the first 100_000 pixels are anomalous
    # which corresponds to the first 100_000 highest scores (2 to 7)
    pred_anom2 = pred_norm.clone()
    mask_anom2 = torch.concatenate([torch.ones(100_000), torch.zeros(900_000)]).reshape(shape).to(torch.int32)
    expected_tpr_anom2 = (10 * expected_fpr_norm).clip(0, 1)

    anomaly_maps = torch.stack([pred_norm, pred_anom1, pred_anom2], axis=0)
    masks = torch.stack([mask_norm, mask_anom1, mask_anom2], axis=0)

    expected_shared_fpr = expected_fpr_norm
    expected_per_image_tprs = torch.stack([expected_tpr_norm, expected_tpr_anom1, expected_tpr_anom2], axis=0)
    expected_image_classes = torch.tensor([0, 1, 1], dtype=torch.int32)

    if metafunc.function is test_pimo or metafunc.function is test_aupimo_values:
        argvalues_tensors = [
            (
                anomaly_maps,
                masks,
                expected_thresholds,
                expected_shared_fpr,
                expected_per_image_tprs,
                expected_image_classes,
            ),
            (
                10 * anomaly_maps,
                masks,
                10 * expected_thresholds,
                expected_shared_fpr,
                expected_per_image_tprs,
                expected_image_classes,
            ),
        ]
        metafunc.parametrize(
            argnames=(
                "anomaly_maps",
                "masks",
                "expected_thresholds",
                "expected_shared_fpr",
                "expected_per_image_tprs",
                "expected_image_classes",
            ),
            argvalues=argvalues_tensors,
        )

    if metafunc.function is test_aupimo_values:
        argvalues_tensors = [
            (
                (1e-1, 1.0),
                torch.tensor(
                    [
                        torch.nan,
                        # recall: trapezium area = (a + b) * h / 2
                        (0.10 + 1.0) * 1 / 2,
                        (1.0 + 1.0) * 1 / 2,
                    ],
                    dtype=torch.float64,
                ),
            ),
            (
                (1e-3, 1e-1),
                torch.tensor(
                    [
                        torch.nan,
                        # average of two trapezium areas / 2 (normalizing factor)
                        (((1e-3 + 1e-2) * 1 / 2) + ((1e-2 + 1e-1) * 1 / 2)) / 2,
                        (((1e-2 + 1e-1) * 1 / 2) + ((1e-1 + 1.0) * 1 / 2)) / 2,
                    ],
                    dtype=torch.float64,
                ),
            ),
            (
                (1e-5, 1e-4),
                torch.tensor(
                    [
                        torch.nan,
                        (1e-5 + 1e-4) * 1 / 2,
                        (1e-4 + 1e-3) * 1 / 2,
                    ],
                    dtype=torch.float64,
                ),
            ),
        ]
        metafunc.parametrize(
            argnames=(
                "fpr_bounds",
                "expected_aupimos",  # trapezoid surfaces
            ),
            argvalues=argvalues_tensors,
        )

    if metafunc.function is test_aupimo_edge:
        metafunc.parametrize(
            argnames=(
                "anomaly_maps",
                "masks",
            ),
            argvalues=[
                (
                    anomaly_maps,
                    masks,
                ),
                (
                    10 * anomaly_maps,
                    masks,
                ),
            ],
        )
        metafunc.parametrize(
            argnames=("fpr_bounds",),
            argvalues=[
                ((1e-1, 1.0),),
                ((1e-3, 1e-2),),
                ((1e-5, 1e-4),),
                (None,),
            ],
        )


def _do_test_pimo_outputs(
    thresholds: Tensor,
    shared_fpr: Tensor,
    per_image_tprs: Tensor,
    image_classes: Tensor,
    expected_thresholds: Tensor,
    expected_shared_fpr: Tensor,
    expected_per_image_tprs: Tensor,
    expected_image_classes: Tensor,
) -> None:
    """Test if the outputs of any of the PIMO interfaces are correct."""
    assert isinstance(shared_fpr, Tensor)
    assert isinstance(per_image_tprs, Tensor)
    assert isinstance(image_classes, Tensor)
    assert isinstance(expected_thresholds, Tensor)
    assert isinstance(expected_shared_fpr, Tensor)
    assert isinstance(expected_per_image_tprs, Tensor)
    assert isinstance(expected_image_classes, Tensor)
    allclose = torch.allclose

    assert thresholds.ndim == 1
    assert shared_fpr.ndim == 1
    assert per_image_tprs.ndim == 2
    assert tuple(image_classes.shape) == (3,)

    assert allclose(thresholds, expected_thresholds)
    assert allclose(shared_fpr, expected_shared_fpr)
    assert allclose(per_image_tprs, expected_per_image_tprs, equal_nan=True)
    assert (image_classes == expected_image_classes).all()


def test_pimo(
    anomaly_maps: Tensor,
    masks: Tensor,
    expected_thresholds: Tensor,
    expected_shared_fpr: Tensor,
    expected_per_image_tprs: Tensor,
    expected_image_classes: Tensor,
) -> None:
    """Test if `pimo()` returns the expected values."""

    def do_assertions(pimo_result: PIMOResult) -> None:
        thresholds = pimo_result.thresholds
        shared_fpr = pimo_result.shared_fpr
        per_image_tprs = pimo_result.per_image_tprs
        image_classes = pimo_result.image_classes
        _do_test_pimo_outputs(
            thresholds,
            shared_fpr,
            per_image_tprs,
            image_classes,
            expected_thresholds,
            expected_shared_fpr,
            expected_per_image_tprs,
            expected_image_classes,
        )

    # metric interface
    metric = pimo.PIMO(
        num_thresholds=7,
    )
    metric.update(anomaly_maps, masks)
    pimo_result = metric.compute()
    do_assertions(pimo_result)


def _do_test_aupimo_outputs(
    thresholds: Tensor,
    shared_fpr: Tensor,
    per_image_tprs: Tensor,
    image_classes: Tensor,
    aupimos: Tensor,
    expected_thresholds: Tensor,
    expected_shared_fpr: Tensor,
    expected_per_image_tprs: Tensor,
    expected_image_classes: Tensor,
    expected_aupimos: Tensor,
) -> None:
    _do_test_pimo_outputs(
        thresholds,
        shared_fpr,
        per_image_tprs,
        image_classes,
        expected_thresholds,
        expected_shared_fpr,
        expected_per_image_tprs,
        expected_image_classes,
    )
    assert isinstance(aupimos, Tensor)
    assert isinstance(expected_aupimos, Tensor)
    allclose = torch.allclose
    assert tuple(aupimos.shape) == (3,)
    assert allclose(aupimos, expected_aupimos, equal_nan=True)


def test_aupimo_values(
    anomaly_maps: torch.Tensor,
    masks: torch.Tensor,
    fpr_bounds: tuple[float, float],
    expected_thresholds: torch.Tensor,
    expected_shared_fpr: torch.Tensor,
    expected_per_image_tprs: torch.Tensor,
    expected_image_classes: torch.Tensor,
    expected_aupimos: torch.Tensor,
) -> None:
    """Test if `aupimo()` returns the expected values."""

    def do_assertions(pimo_result: PIMOResult, aupimo_result: AUPIMOResult) -> None:
        # test metadata
        assert aupimo_result.fpr_bounds == fpr_bounds
        # recall: this one is not the same as the number of thresholds in the curve
        # this is the number of thresholds used to compute the integral in `aupimo()`
        # always less because of the integration bounds
        assert aupimo_result.num_thresholds < 7

        # test data
        # from pimo result
        thresholds = pimo_result.thresholds
        shared_fpr = pimo_result.shared_fpr
        per_image_tprs = pimo_result.per_image_tprs
        image_classes = pimo_result.image_classes
        # from aupimo result
        aupimos = aupimo_result.aupimos
        _do_test_aupimo_outputs(
            thresholds,
            shared_fpr,
            per_image_tprs,
            image_classes,
            aupimos,
            expected_thresholds,
            expected_shared_fpr,
            expected_per_image_tprs,
            expected_image_classes,
            expected_aupimos,
        )
        thresh_lower_bound = aupimo_result.thresh_lower_bound
        thresh_upper_bound = aupimo_result.thresh_upper_bound
        assert anomaly_maps.min() <= thresh_lower_bound < thresh_upper_bound <= anomaly_maps.max()

    # metric interface
    metric = pimo.AUPIMO(
        num_thresholds=7,
        fpr_bounds=fpr_bounds,
        return_average=False,
        force=True,
    )
    metric.update(anomaly_maps, masks)
    pimo_result_from_metric, aupimo_result_from_metric = metric.compute()
    do_assertions(pimo_result_from_metric, aupimo_result_from_metric)

    # metric interface
    metric = pimo.AUPIMO(
        num_thresholds=7,
        fpr_bounds=fpr_bounds,
        return_average=True,  # only return the average AUPIMO
        force=True,
    )
    metric.update(anomaly_maps, masks)
    metric.compute()


def test_aupimo_edge(
    anomaly_maps: torch.Tensor,
    masks: torch.Tensor,
    fpr_bounds: tuple[float, float],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test some edge cases."""
    # None is the case of testing the default bounds
    fpr_bounds = {"fpr_bounds": fpr_bounds} if fpr_bounds is not None else {}

    # not enough points on the curve
    # 10 thresholds / 6 decades = 1.6 thresholds per decade < 3
    with pytest.raises(RuntimeError):  # force=False --> raise error
        functional.aupimo_scores(
            anomaly_maps,
            masks,
            num_thresholds=10,
            force=False,
            **fpr_bounds,
        )

    with caplog.at_level(logging.WARNING):  # force=True --> warn
        functional.aupimo_scores(
            anomaly_maps,
            masks,
            num_thresholds=10,
            force=True,
            **fpr_bounds,
        )
    assert "Computation was forced!" in caplog.text

    # default number of points on the curve (300k thresholds) should be enough
    torch.manual_seed(42)
    functional.aupimo_scores(
        anomaly_maps * torch.FloatTensor(anomaly_maps.shape).uniform_(1.0, 1.1),
        masks,
        force=False,
        **fpr_bounds,
    )
