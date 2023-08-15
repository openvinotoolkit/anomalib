"""Tests for the Per-Image Binary Classification Curve (PerImageBinClfCurve) metric."""

import numpy as np
import pytest
import torch

from anomalib.utils.metrics.perimg.binclf_curve import (
    PerImageBinClfCurve,
    __binclf_curves_ndarray_itertools,
    _binclf_curves_ndarray_itertools,
    _perimg_binclf_curve_compute_cpu,
)


def pytest_generate_tests(metafunc):
    pred = np.arange(4, dtype=np.float32)
    thresholds = np.arange(4, dtype=np.float32)

    mask_norm = np.zeros(4).astype(bool)
    expected_norm = np.stack(
        [
            np.array([[0, 4], [0, 0]]),
            np.array([[1, 3], [0, 0]]),
            np.array([[2, 2], [0, 0]]),
            np.array([[3, 1], [0, 0]]),
        ],
        axis=0,
    ).astype(int)

    mask_anom = np.concatenate([np.zeros(2), np.ones(2)]).astype(bool)
    expected_anom = np.stack(
        [
            np.array([[0, 2], [0, 2]]),
            np.array([[1, 1], [0, 2]]),
            np.array([[2, 0], [0, 2]]),
            np.array([[2, 0], [1, 1]]),
        ],
        axis=0,
    ).astype(int)

    if metafunc.function is test___binclf_curves_ndarray_itertools:
        metafunc.parametrize(
            argnames=("pred", "mask", "thresholds", "expected"),
            argvalues=[
                (pred, mask_anom, thresholds[:3], expected_anom[:3]),
                (pred, mask_anom, thresholds, expected_anom),
                (pred, mask_norm, thresholds, expected_norm),
            ],
        )

    preds = np.stack([pred, pred], axis=0)
    masks = np.stack([mask_anom, mask_norm], axis=0)
    expecteds = np.stack([expected_anom, expected_norm], axis=0)

    if metafunc.function is test__binclf_curves_ndarray_itertools:
        metafunc.parametrize(
            argnames=("preds", "masks", "thresholds", "expecteds"),
            argvalues=[
                (preds, masks, thresholds[:3], expecteds[:, :3]),
                (preds, masks, thresholds, expecteds),
            ],
        )

    preds = torch.from_numpy(preds)
    masks = torch.from_numpy(masks)
    thresholds = torch.from_numpy(thresholds)
    expecteds = torch.from_numpy(expecteds)

    if metafunc.function is test__perimg_binclf_curve_compute_cpu:
        metafunc.parametrize(
            argnames=("preds", "masks", "expected_thresholds", "expecteds"),
            argvalues=[
                (preds[:1], masks[:1], thresholds, expecteds[:1]),
                (preds, masks, thresholds, expecteds),
                (preds.reshape(2, 2, 2), masks.reshape(2, 2, 2), thresholds, expecteds),
            ],
        )

    images_classes = torch.tensor([1, 0])

    if metafunc.function is test_perimgbinclfcurve_class:
        metafunc.parametrize(
            argnames=("preds", "masks", "expected_images_classes", "expected_thresholds", "expecteds"),
            argvalues=[
                (preds, masks, images_classes, thresholds[:4], expecteds),
                (preds.reshape(2, 2, 2), masks.reshape(2, 2, 2), images_classes, thresholds, expecteds),
            ],
        )


# with double `_`
def test___binclf_curves_ndarray_itertools(pred, mask, thresholds, expected):
    computed = __binclf_curves_ndarray_itertools(pred, mask, thresholds)
    assert computed.shape == (thresholds.size, 2, 2)
    assert (computed == expected).all()


# with single `_`
def test__binclf_curves_ndarray_itertools(preds, masks, thresholds, expecteds):
    computed = _binclf_curves_ndarray_itertools(preds, masks, thresholds)
    assert computed.shape == (preds.shape[0], thresholds.size, 2, 2)
    assert (computed == expecteds).all()


def test____binclf_curves_ndarray_itertools_validations():
    # `pred` and `mask` must have the same length
    with pytest.raises(ValueError):
        __binclf_curves_ndarray_itertools(np.arange(4), np.arange(3), np.arange(5))

    # `pred` and `mask` must be 1D
    with pytest.raises(ValueError):
        __binclf_curves_ndarray_itertools(np.arange(4).reshape(2, 2), np.arange(4), np.arange(5))

    # `thresholds` must be 1D
    with pytest.raises(ValueError):
        __binclf_curves_ndarray_itertools(np.arange(4), np.arange(4), np.arange(6).reshape(2, 3))


def test__perimg_binclf_curve_compute_cpu(preds, masks, expected_thresholds, expecteds):
    th_bounds = torch.tensor((expected_thresholds[0], expected_thresholds[-1]))

    computed_thresholds, computed = _perimg_binclf_curve_compute_cpu(
        preds, masks, th_bounds, expected_thresholds.numel()
    )
    assert computed.shape == expecteds.shape
    assert (computed == expecteds).all()
    assert (computed_thresholds == expected_thresholds).all()

    # preds and masks can have any shape as long as they are the same
    _perimg_binclf_curve_compute_cpu(
        preds.reshape(-1, 2, 2), masks.reshape(-1, 2, 2), th_bounds, expected_thresholds.numel()
    )
    _perimg_binclf_curve_compute_cpu(
        preds.reshape(2, 1, 1, -1), masks.reshape(2, 1, 1, -1), th_bounds, expected_thresholds.numel()
    )

    # it's ok to have the threhsholds beyond the range of the preds
    _perimg_binclf_curve_compute_cpu(preds, masks, th_bounds * 2, expected_thresholds.numel())

    # or inside the bounds without reaching them
    _perimg_binclf_curve_compute_cpu(preds, masks, th_bounds * 0.5, expected_thresholds.numel())

    # it's also ok to have more thresholds than unique values in the preds
    _perimg_binclf_curve_compute_cpu(preds, masks, th_bounds, expected_thresholds.numel() * 2)

    # or less
    _perimg_binclf_curve_compute_cpu(preds, masks, th_bounds, expected_thresholds.numel() // 2)


def test__perimg_binclf_curve_compute_cpu_validations():
    preds_ok = torch.from_numpy(np.arange(4, dtype=np.float32))
    preds_ok = torch.stack([preds_ok, preds_ok], axis=0)

    masks_ok = torch.from_numpy(np.concatenate([np.zeros(2), np.ones(2)]).astype(bool))
    masks_ok = torch.stack([masks_ok, masks_ok], axis=0)

    th_bounds_ok = torch.tensor((0, 3))

    num_ths_ok = 5

    # `num_thresholds` must be > 1
    with pytest.raises(ValueError):
        _perimg_binclf_curve_compute_cpu(preds_ok, masks_ok, th_bounds_ok, 1)

    # `num_thresholds` must be an integer
    with pytest.raises(ValueError):
        _perimg_binclf_curve_compute_cpu(preds_ok, masks_ok, th_bounds_ok, 1.5)

    # `th_bounds` must be lower-than-higher
    with pytest.raises(ValueError):
        _perimg_binclf_curve_compute_cpu(preds_ok, masks_ok, th_bounds_ok.flip(0), num_ths_ok)

    # `masks` and `preds` must have the same shape
    with pytest.raises(ValueError):
        _perimg_binclf_curve_compute_cpu(preds_ok[:1], masks_ok, th_bounds_ok, num_ths_ok)

    with pytest.raises(ValueError):
        _perimg_binclf_curve_compute_cpu(preds_ok, masks_ok.reshape(2, 2, 2), th_bounds_ok, num_ths_ok)

    if torch.cuda.is_available():
        # `preds` and `masks` must both be in the cpu
        with pytest.raises(ValueError):
            _perimg_binclf_curve_compute_cpu(preds_ok.to("cuda"), masks_ok, th_bounds_ok, num_ths_ok)
        with pytest.raises(ValueError):
            _perimg_binclf_curve_compute_cpu(preds_ok, masks_ok.to("cuda"), th_bounds_ok, num_ths_ok)


def test_perimgbinclfcurve_class(preds, masks, expected_images_classes, expected_thresholds, expecteds):
    num_ths = expected_thresholds.numel()

    metric = PerImageBinClfCurve(num_thresholds=num_ths)
    _ = metric.compute()  # should not break

    metric.update(preds, masks)
    computed_thresholds, computed, computed_images_classes = metric.compute()

    assert computed_thresholds.shape == expected_thresholds.shape
    assert computed.shape == expecteds.shape
    assert computed_images_classes.shape == expected_images_classes.shape

    assert (computed_thresholds == expected_thresholds).all()
    assert (computed == expecteds).all()
    assert (computed_images_classes == expected_images_classes).all()

    # just execute
    tprs = metric.tprs(computed)
    fprs = metric.fprs(computed)

    assert tprs.ndim == 1
    assert fprs.ndim == 1
