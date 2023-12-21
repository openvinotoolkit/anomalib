"""Tests for per-image binary classification curves using numpy and numba versions."""
# ruff: noqa: SLF001, PT011

import numpy as np
import pytest
import torch
from numpy import ndarray
from torch import Tensor


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Generate test cases."""
    pred = np.arange(1, 5, dtype=np.float32)
    threshs = np.arange(1, 5, dtype=np.float32)

    gt_norm = np.zeros(4).astype(bool)
    gt_anom = np.concatenate([np.zeros(2), np.ones(2)]).astype(bool)

    # in the case where thresholds are all unique values in the predictions
    expected_norm = np.stack(
        [
            np.array([[0, 4], [0, 0]]),
            np.array([[1, 3], [0, 0]]),
            np.array([[2, 2], [0, 0]]),
            np.array([[3, 1], [0, 0]]),
        ],
        axis=0,
    ).astype(int)
    expected_anom = np.stack(
        [
            np.array([[0, 2], [0, 2]]),
            np.array([[1, 1], [0, 2]]),
            np.array([[2, 0], [0, 2]]),
            np.array([[2, 0], [1, 1]]),
        ],
        axis=0,
    ).astype(int)

    # in the case where all thresholds are higher than the highest prediction
    expected_norm_threshs_too_high = np.stack(
        [
            np.array([[4, 0], [0, 0]]),
            np.array([[4, 0], [0, 0]]),
            np.array([[4, 0], [0, 0]]),
            np.array([[4, 0], [0, 0]]),
        ],
        axis=0,
    ).astype(int)
    expected_anom_threshs_too_high = np.stack(
        [
            np.array([[2, 0], [2, 0]]),
            np.array([[2, 0], [2, 0]]),
            np.array([[2, 0], [2, 0]]),
            np.array([[2, 0], [2, 0]]),
        ],
        axis=0,
    ).astype(int)

    # in the case where all thresholds are lower than the lowest prediction
    expected_norm_threshs_too_low = np.stack(
        [
            np.array([[0, 4], [0, 0]]),
            np.array([[0, 4], [0, 0]]),
            np.array([[0, 4], [0, 0]]),
            np.array([[0, 4], [0, 0]]),
        ],
        axis=0,
    ).astype(int)
    expected_anom_threshs_too_low = np.stack(
        [
            np.array([[0, 2], [0, 2]]),
            np.array([[0, 2], [0, 2]]),
            np.array([[0, 2], [0, 2]]),
            np.array([[0, 2], [0, 2]]),
        ],
        axis=0,
    ).astype(int)

    if metafunc.function is test__binclf_one_curve_python or metafunc.function is test__binclf_one_curve_numba:
        metafunc.parametrize(
            argnames=("pred", "gt", "threshs", "expected"),
            argvalues=[
                (pred, gt_anom, threshs[:3], expected_anom[:3]),
                (pred, gt_anom, threshs, expected_anom),
                (pred, gt_norm, threshs, expected_norm),
                (pred, gt_norm, 10 * threshs, expected_norm_threshs_too_high),
                (pred, gt_anom, 10 * threshs, expected_anom_threshs_too_high),
                (pred, gt_norm, 0.001 * threshs, expected_norm_threshs_too_low),
                (pred, gt_anom, 0.001 * threshs, expected_anom_threshs_too_low),
            ],
        )

    preds = np.stack([pred, pred], axis=0)
    gts = np.stack([gt_anom, gt_norm], axis=0)
    binclf_curves = np.stack([expected_anom, expected_norm], axis=0)
    binclf_curves_threshs_too_high = np.stack([expected_anom_threshs_too_high, expected_norm_threshs_too_high], axis=0)
    binclf_curves_threshs_too_low = np.stack([expected_anom_threshs_too_low, expected_norm_threshs_too_low], axis=0)

    if (
        metafunc.function is test__binclf_multiple_curves_python
        or metafunc.function is test__binclf_multiple_curves_numba
    ):
        metafunc.parametrize(
            argnames=("preds", "gts", "threshs", "expecteds"),
            argvalues=[
                (preds, gts, threshs[:3], binclf_curves[:, :3]),
                (preds, gts, threshs, binclf_curves),
            ],
        )

    if metafunc.function is test_binclf_multiple_curves:
        metafunc.parametrize(
            argnames=(
                "preds",
                "gts",
                "threshs",
                "expected_binclf_curves",
            ),
            argvalues=[
                (preds[:1], gts[:1], threshs, binclf_curves[:1]),
                (preds, gts, threshs, binclf_curves),
                (10 * preds, gts, 10 * threshs, binclf_curves),
            ],
        )
        metafunc.parametrize(
            argnames=("algorithm",),
            argvalues=[
                ("python",),
                ("numba",),
            ],
        )

    if metafunc.function is test_binclf_multiple_curves_validations:
        metafunc.parametrize(
            argnames=("args", "exception"),
            argvalues=[
                # `scores` and `gts` must be 2D
                ([preds.reshape(2, 2, 2), gts, threshs], ValueError),
                ([preds, gts.flatten(), threshs], ValueError),
                # `threshs` must be 1D
                ([preds, gts, threshs.reshape(2, 2)], ValueError),
                # `scores` and `gts` must have the same shape
                ([preds, gts[:1], threshs], ValueError),
                ([preds[:, :2], gts, threshs], ValueError),
                # `scores` be of type float
                ([preds.astype(int), gts, threshs], TypeError),
                # `gts` be of type bool
                ([preds, gts.astype(int), threshs], TypeError),
                # `threshs` be of type float
                ([preds, gts, threshs.astype(int)], TypeError),
                # `threshs` must be sorted in ascending order
                ([preds, gts, np.flip(threshs)], ValueError),
                ([preds, gts, np.concatenate([threshs[-2:], threshs[:2]])], ValueError),
            ],
        )
        metafunc.parametrize(
            argnames=("kwargs",),
            argvalues=[
                ({"algorithm": "python"},),
                ({"algorithm": "numba"},),
            ],
        )

    # the following tests are for `per_img_binclf_curve()`, which expects
    # inputs in image spatial format, i.e. (height, width)
    preds = preds.reshape(2, 2, 2)
    gts = gts.reshape(2, 2, 2)

    per_img_binclf_curves_numpy_argvalues = [
        # `threshs_choice` = "given"
        (
            preds,
            gts,
            "given",
            threshs,
            None,
            threshs,
            binclf_curves,
        ),
        (
            preds,
            gts,
            "given",
            10 * threshs,
            2,
            10 * threshs,
            binclf_curves_threshs_too_high,
        ),
        (
            preds,
            gts,
            "given",
            0.01 * threshs,
            None,
            0.01 * threshs,
            binclf_curves_threshs_too_low,
        ),
        # `threshs_choice` = 'minmax-linspace'"
        (
            preds,
            gts,
            "minmax-linspace",
            None,
            len(threshs),
            threshs,
            binclf_curves,
        ),
        (
            2 * preds,
            gts.astype(int),  # this is ok
            "minmax-linspace",
            None,
            len(threshs),
            2 * threshs,
            binclf_curves,
        ),
    ]

    if metafunc.function is test_per_img_binclf_curve_numpy:
        metafunc.parametrize(
            argnames=(
                "anomaly_maps",
                "masks",
                "threshs_choice",
                "threshs_given",
                "num_threshs",
                "expected_threshs",
                "expected_binclf_curves",
            ),
            argvalues=per_img_binclf_curves_numpy_argvalues,
        )

    # the test with the torch interface are the same we just convert ndarray to Tensor
    if metafunc.function is test_per_img_binclf_curve_torch:
        metafunc.parametrize(
            argnames=(
                "anomaly_maps",
                "masks",
                "threshs_choice",
                "threshs_given",
                "num_threshs",
                "expected_threshs",
                "expected_binclf_curves",
            ),
            argvalues=[
                tuple(torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in argvals)
                for argvals in per_img_binclf_curves_numpy_argvalues
            ],
        )

    if metafunc.function is test_per_img_binclf_curve_numpy or metafunc.function is test_per_img_binclf_curve_torch:
        metafunc.parametrize(
            argnames=("algorithm",),
            argvalues=[
                ("python",),
                ("numba",),
            ],
        )

    # if metafunc.function is test_per_img_binclf_curve_numpy:

    if metafunc.function is test_per_img_binclf_curve_numpy_validations:
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
                ([preds.astype(int), gts], TypeError),
                # `gts` be of type bool or int
                ([preds, gts.astype(float)], TypeError),
                # `threshs` be of type float
                ([preds, gts, threshs.astype(int)], TypeError),
            ],
        )
        metafunc.parametrize(
            argnames=("kwargs",),
            argvalues=[
                ({"algorithm": "numba", "threshs_choice": "given", "threshs_given": threshs, "num_threshs": None},),
                (
                    {
                        "algorithm": "python",
                        "threshs_choice": "minmax-linspace",
                        "threshs_given": None,
                        "num_threshs": len(threshs),
                    },
                ),
            ],
        )


# ==================================================================================================
# LOW-LEVEL FUNCTIONS (PYTHON)


def test__binclf_one_curve_python(pred: ndarray, gt: ndarray, threshs: ndarray, expected: ndarray) -> None:
    """Test if `_binclf_one_curve_python()` returns the expected values."""
    from anomalib.metrics.per_image import binclf_curve_numpy

    computed = binclf_curve_numpy._binclf_one_curve_python(pred, gt, threshs)
    assert computed.shape == (threshs.size, 2, 2)
    assert (computed == expected).all()


def test__binclf_multiple_curves_python(
    preds: ndarray,
    gts: ndarray,
    threshs: ndarray,
    expecteds: ndarray,
) -> None:
    """Test if `_binclf_multiple_curves_python()` returns the expected values."""
    from anomalib.metrics.per_image import binclf_curve_numpy

    computed = binclf_curve_numpy._binclf_multiple_curves_python(preds, gts, threshs)
    assert computed.shape == (preds.shape[0], threshs.size, 2, 2)
    assert (computed == expecteds).all()


# ==================================================================================================
# LOW-LEVEL FUNCTIONS (NUMBA)


def test__binclf_one_curve_numba(pred: ndarray, gt: ndarray, threshs: ndarray, expected: ndarray) -> None:
    """Test if `_binclf_one_curve_numba()` returns the expected values."""
    from anomalib.metrics.per_image import _binclf_curve_numba

    computed = _binclf_curve_numba.binclf_one_curve_numba(pred, gt, threshs)
    assert computed.shape == (threshs.size, 2, 2)
    assert (computed == expected).all()


def test__binclf_multiple_curves_numba(preds: ndarray, gts: ndarray, threshs: ndarray, expecteds: ndarray) -> None:
    """Test if `_binclf_multiple_curves_python()` returns the expected values."""
    from anomalib.metrics.per_image import _binclf_curve_numba

    computed = _binclf_curve_numba.binclf_multiple_curves_numba(preds, gts, threshs)
    assert computed.shape == (preds.shape[0], threshs.size, 2, 2)
    assert (computed == expecteds).all()


# ==================================================================================================
# API FUNCTIONS (NUMPY)


def test_binclf_multiple_curves(
    preds: ndarray,
    gts: ndarray,
    threshs: ndarray,
    expected_binclf_curves: ndarray,
    algorithm: str,
) -> None:
    """Test if `binclf_multiple_curves()` returns the expected values."""
    from anomalib.metrics.per_image import binclf_curve_numpy

    computed = binclf_curve_numpy.binclf_multiple_curves(
        preds,
        gts,
        threshs,
        algorithm=algorithm,
    )
    assert computed.shape == expected_binclf_curves.shape
    assert (computed == expected_binclf_curves).all()

    # it's ok to have the threhsholds beyond the range of the preds
    binclf_curve_numpy.binclf_multiple_curves(preds, gts, 2 * threshs, algorithm=algorithm)

    # or inside the bounds without reaching them
    binclf_curve_numpy.binclf_multiple_curves(preds, gts, 0.5 * threshs, algorithm=algorithm)

    # it's also ok to have more threshs than unique values in the preds
    # add the values in between the threshs
    threshs_unncessary = 0.5 * (threshs[:-1] + threshs[1:])
    threshs_unncessary = np.concatenate([threshs_unncessary, threshs])
    threshs_unncessary = np.sort(threshs_unncessary)
    binclf_curve_numpy.binclf_multiple_curves(preds, gts, threshs_unncessary, algorithm=algorithm)

    # or less
    binclf_curve_numpy.binclf_multiple_curves(preds, gts, threshs[1:3], algorithm=algorithm)


def test_binclf_multiple_curves_validations(args: list, kwargs: dict, exception: Exception) -> None:
    """Test if `_binclf_multiple_curves_python()` raises the expected errors."""
    from anomalib.metrics.per_image import binclf_curve_numpy

    with pytest.raises(exception):
        binclf_curve_numpy.binclf_multiple_curves(*args, **kwargs)


def test_per_img_binclf_curve_numpy(
    anomaly_maps: ndarray,
    masks: ndarray,
    algorithm: str,
    threshs_choice: str,
    threshs_given: ndarray | None,
    num_threshs: int | None,
    expected_threshs: ndarray,
    expected_binclf_curves: ndarray,
) -> None:
    """Test if `per_img_binclf_curve()` returns the expected values."""
    from anomalib.metrics.per_image import binclf_curve_numpy

    computed_threshs, computed_binclf_curves = binclf_curve_numpy.per_img_binclf_curve(
        anomaly_maps,
        masks,
        algorithm=algorithm,
        threshs_choice=threshs_choice,
        threshs_given=threshs_given,
        num_threshs=num_threshs,
    )

    # threshs
    assert computed_threshs.shape == expected_threshs.shape
    assert computed_threshs.dtype == computed_threshs.dtype
    assert (computed_threshs == expected_threshs).all()

    # binclf_curves
    assert computed_binclf_curves.shape == expected_binclf_curves.shape
    assert computed_binclf_curves.dtype == expected_binclf_curves.dtype
    assert (computed_binclf_curves == expected_binclf_curves).all()


def test_per_img_binclf_curve_numpy_validations(args: list, kwargs: dict, exception: Exception) -> None:
    """Test if `per_img_binclf_curve()` raises the expected errors."""
    from anomalib.metrics.per_image import binclf_curve_numpy

    with pytest.raises(exception):
        binclf_curve_numpy.per_img_binclf_curve(*args, **kwargs)


# ==================================================================================================
# API FUNCTIONS (TORCH)


def test_per_img_binclf_curve_torch(
    anomaly_maps: Tensor,
    masks: Tensor,
    algorithm: str,
    threshs_choice: str,
    threshs_given: Tensor | None,
    num_threshs: int | None,
    expected_threshs: Tensor,
    expected_binclf_curves: Tensor,
) -> None:
    """Test if `per_img_binclf_curve()` returns the expected values."""
    from anomalib.metrics.per_image import binclf_curve

    results_object = binclf_curve.per_img_binclf_curve(
        anomaly_maps,
        masks,
        algorithm=algorithm,
        threshs_choice=threshs_choice,
        threshs_given=threshs_given,
        num_threshs=num_threshs,
    )
    computed_threshs, computed_binclf_curves = results_object.threshs, results_object.binclf_curves

    # threshs
    assert computed_threshs.shape == expected_threshs.shape
    assert computed_threshs.dtype == computed_threshs.dtype
    assert (computed_threshs == expected_threshs).all()

    # binclf_curves
    assert computed_binclf_curves.shape == expected_binclf_curves.shape
    assert computed_binclf_curves.dtype == expected_binclf_curves.dtype
    assert (computed_binclf_curves == expected_binclf_curves).all()

    # test properties
    results_object.fprs  # noqa: B018
    results_object.tprs  # noqa: B018
