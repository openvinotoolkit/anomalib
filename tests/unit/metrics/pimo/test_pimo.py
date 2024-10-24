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
    shape = (1000, 1000)  # (H, W), 1 million pixels

    # --- normal ---
    # histogram of scores:
    # value:   7   6    5    4    3    2     1
    # count:   1   9   90  900   9k   90k  900k
    # cumsum:  1  10  100   1k  10k  100k    1M
    # proportion (1e{})
    #         -6  -5   -4   -3   -2    -1     0
    pred_norm = torch.ones(1_000_000, dtype=torch.float32)
    pred_norm[:100_000] += 1
    pred_norm[:10_000] += 1
    pred_norm[:1_000] += 1
    pred_norm[:100] += 1
    pred_norm[:10] += 1
    pred_norm[:1] += 1
    pred_norm = pred_norm.reshape(shape)
    mask_norm = torch.zeros_like(pred_norm, dtype=torch.int32)

    # --- anomalous ---
    pred_anom1 = pred_norm.clone()
    mask_anom1 = torch.ones_like(pred_anom1, dtype=torch.int32)

    # only the first 100_000 pixels are anomalous
    # which corresponds to the first 100_000 highest scores (2 to 7)
    pred_anom2 = pred_norm.clone()
    mask_anom2 = torch.concatenate([torch.ones(100_000), torch.zeros(900_000)]).reshape(shape).to(torch.int32)

    anomaly_maps = torch.stack([pred_norm, pred_anom1, pred_anom2], axis=0)
    masks = torch.stack([mask_norm, mask_anom1, mask_anom2], axis=0)

    if metafunc.function is test_pimo or metafunc.function is test_aupimo or metafunc.function is test_aupimo_edge:
        metafunc.parametrize(argnames=("anomaly_maps", "masks"), argvalues=[(anomaly_maps, masks)])

    if metafunc.function is test_aupimo:
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

    # === random values ===
    generator = torch.Generator().manual_seed(42)
    masks_normals = torch.zeros((6, 1024, 1024), dtype=torch.int32)
    anomaly_maps_normals = torch.normal(0, 1, (6, 1024, 1024), generator=generator)
    masks_anomalous = torch.zeros_like(masks_normals)
    # make some pixels anomalous
    masks_anomalous[0, 512:, 512:] = 1
    masks_anomalous[1, :512, :512] = 1
    masks_anomalous[2, :512, 512:] = 1
    masks_anomalous[3, 512:, :512] = 1
    masks_anomalous[4, 256:768, 256:768] = 1
    masks_anomalous[5, 256:768, 256:768] = 1
    anomaly_maps_anomalous = torch.where(
        masks_anomalous.bool(),
        torch.normal(1, 1, (6, 1024, 1024), generator=generator),
        torch.normal(0, 1, (6, 1024, 1024), generator=generator),
    )
    anomaly_maps = torch.concatenate([anomaly_maps_normals, anomaly_maps_anomalous], axis=0)
    masks = torch.concatenate([masks_normals, masks_anomalous], axis=0)

    if metafunc.function is test_pimo_random_values or metafunc.function is test_aupimo_random_values:
        metafunc.parametrize(argnames=("anomaly_maps", "masks"), argvalues=[(anomaly_maps, masks)])


def test_pimo_random_values(anomaly_maps: Tensor, masks: Tensor) -> None:
    """Make sure the function runs without errors, types and shapes are correct."""
    # metric interface
    metric = pimo.PIMO(fpr_bounds=(1e-5, 1e-3), num_thresholds=300)
    metric.update(anomaly_maps, masks)
    pimo_result: PIMOResult = metric.compute()

    assert isinstance(pimo_result.thresholds, Tensor)
    assert pimo_result.thresholds.ndim == 1
    assert pimo_result.thresholds.shape == (300,)

    assert isinstance(pimo_result.shared_fpr, Tensor)
    assert pimo_result.shared_fpr.ndim == 1
    assert pimo_result.shared_fpr.shape == (300,)

    assert isinstance(pimo_result.per_image_tprs, Tensor)
    assert pimo_result.per_image_tprs.ndim == 2
    assert pimo_result.per_image_tprs.shape == (12, 300)

    assert isinstance(pimo_result.image_classes, Tensor)
    assert pimo_result.image_classes.shape == (12,)

    fpr_upper_bound_defacto = pimo_result.shared_fpr[0]
    assert torch.isclose(fpr_upper_bound_defacto, torch.tensor(1e-3, dtype=torch.float64), rtol=1e-3)

    fpr_lower_bound_defacto = pimo_result.shared_fpr[-1]
    assert torch.isclose(fpr_lower_bound_defacto, torch.tensor(1e-5, dtype=torch.float64), rtol=1e-3)


def test_aupimo_random_values(anomaly_maps: Tensor, masks: Tensor) -> None:
    """Make sure the function runs without errors, types and shapes are correct."""
    # metric interface
    metric = pimo.AUPIMO(
        fpr_bounds=(1e-5, 1e-3),
        num_thresholds=300,
        return_average=False,
        force=False,
    )
    metric.update(anomaly_maps, masks)
    aupimo_result: AUPIMOResult
    _, aupimo_result = metric.compute()

    assert aupimo_result.fpr_bounds == (1e-5, 1e-3)

    assert aupimo_result.thresh_lower_bound < aupimo_result.thresh_upper_bound
    assert anomaly_maps.min() < aupimo_result.thresh_lower_bound < aupimo_result.thresh_upper_bound < anomaly_maps.max()

    assert isinstance(aupimo_result.aupimos, Tensor)
    assert aupimo_result.aupimos.ndim == 1
    assert aupimo_result.aupimos.shape == (12,)


def _assert_pimo_result_close_to_expected(
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
    assert torch.allclose(thresholds, expected_thresholds, atol=1e-2)
    assert torch.allclose(shared_fpr, expected_shared_fpr)
    assert torch.allclose(per_image_tprs, expected_per_image_tprs, equal_nan=True)
    assert (image_classes == expected_image_classes).all()


def test_pimo(anomaly_maps: Tensor, masks: Tensor) -> None:
    """Test if `pimo()` returns the expected values."""
    # metric interface
    metric = pimo.PIMO(fpr_bounds=(1e-5, 1e-3), num_thresholds=3)
    metric.update(anomaly_maps, masks)
    pimo_result: PIMOResult = metric.compute()
    _assert_pimo_result_close_to_expected(
        thresholds=pimo_result.thresholds,
        shared_fpr=pimo_result.shared_fpr,
        per_image_tprs=pimo_result.per_image_tprs,
        image_classes=pimo_result.image_classes,
        expected_thresholds=torch.tensor([4, 5, 6], dtype=torch.float32),
        expected_shared_fpr=torch.tensor([1e-3, 1e-4, 1e-5], dtype=torch.float64),
        expected_per_image_tprs=torch.stack(
            [
                torch.full((3,), torch.nan, dtype=torch.float64),
                torch.tensor([1e-3, 1e-4, 1e-5], dtype=torch.float64),
                torch.tensor([1e-2, 1e-3, 1e-4], dtype=torch.float64),
            ],
            axis=0,
        ),
        expected_image_classes=torch.tensor([0, 1, 1], dtype=torch.int32),
    )

    # multiplying all scores by a factor should not change the results, only the thresholds
    metric = pimo.PIMO(fpr_bounds=(1e-5, 1e-3), num_thresholds=3)
    metric.update(10 * anomaly_maps, masks)  # x10 anomaly maps
    pimo_result_x10: PIMOResult = metric.compute()
    _assert_pimo_result_close_to_expected(
        thresholds=pimo_result_x10.thresholds,
        shared_fpr=pimo_result_x10.shared_fpr,
        per_image_tprs=pimo_result_x10.per_image_tprs,
        image_classes=pimo_result_x10.image_classes,
        # x10 as well
        expected_thresholds=torch.tensor([40, 50, 60], dtype=torch.float32),
        # all other values are the same
        expected_shared_fpr=torch.tensor([1e-3, 1e-4, 1e-5], dtype=torch.float64),
        expected_per_image_tprs=torch.stack(
            [
                torch.full((3,), torch.nan, dtype=torch.float64),
                torch.tensor([1e-3, 1e-4, 1e-5], dtype=torch.float64),
                torch.tensor([1e-2, 1e-3, 1e-4], dtype=torch.float64),
            ],
            axis=0,
        ),
        expected_image_classes=torch.tensor([0, 1, 1], dtype=torch.int32),
    )

    # different bounds with more thresholds
    metric = pimo.PIMO(fpr_bounds=(1e-5, 1e-2), num_thresholds=7)
    metric.update(anomaly_maps, masks)
    pimo_result_diff_bounds: PIMOResult = metric.compute()
    _assert_pimo_result_close_to_expected(
        thresholds=pimo_result_diff_bounds.thresholds,
        shared_fpr=pimo_result_diff_bounds.shared_fpr,
        per_image_tprs=pimo_result_diff_bounds.per_image_tprs,
        image_classes=pimo_result_diff_bounds.image_classes,
        expected_thresholds=torch.tensor([3, 3.5, 4, 4.5, 5, 5.5, 6], dtype=torch.float32),
        expected_shared_fpr=torch.tensor([1e-2, 1e-3, 1e-3, 1e-4, 1e-4, 1e-5, 1e-5], dtype=torch.float64),
        expected_per_image_tprs=torch.stack(
            [
                torch.full((7,), torch.nan, dtype=torch.float64),
                torch.tensor([1e-2, 1e-3, 1e-3, 1e-4, 1e-4, 1e-5, 1e-5], dtype=torch.float64),
                torch.tensor([1e-1, 1e-2, 1e-2, 1e-3, 1e-3, 1e-4, 1e-4], dtype=torch.float64),
            ],
            axis=0,
        ),
        expected_image_classes=torch.tensor([0, 1, 1], dtype=torch.int32),
    )


def test_aupimo(
    anomaly_maps: torch.Tensor,
    masks: torch.Tensor,
    fpr_bounds: tuple[float, float],
    expected_aupimos: torch.Tensor,
) -> None:
    """Test if `aupimo()` returns the expected values."""
    # metric interface
    metric = pimo.AUPIMO(
        num_thresholds=7,
        fpr_bounds=fpr_bounds,
        return_average=False,
        force=True,
    )
    metric.update(anomaly_maps, masks)
    aupimo_result: AUPIMOResult
    _, aupimo_result = metric.compute()
    torch.allclose(aupimo_result.aupimos, expected_aupimos, equal_nan=True)

    # metric interface
    metric = pimo.AUPIMO(
        num_thresholds=7,
        fpr_bounds=fpr_bounds,
        return_average=True,  # only return the average AUPIMO
        force=True,
    )
    metric.update(anomaly_maps, masks)
    average_aupimo = metric.compute()
    assert torch.allclose(average_aupimo, expected_aupimos[~torch.isnan(expected_aupimos)].mean(), equal_nan=True)


def test_aupimo_edge(
    anomaly_maps: torch.Tensor,
    masks: torch.Tensor,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test some edge cases."""
    # not enough points on the curve
    # force=False --> raise error
    with pytest.raises(RuntimeError):
        functional.aupimo_scores(
            anomaly_maps,
            masks,
            num_thresholds=10,
            force=False,
        )
    # force=True --> warn and compute anyway
    with caplog.at_level(logging.WARNING):
        functional.aupimo_scores(
            anomaly_maps,
            masks,
            num_thresholds=10,
            force=True,
        )
    assert "Computation was forced!" in caplog.text
