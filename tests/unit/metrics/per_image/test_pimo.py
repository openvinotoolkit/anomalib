"""Test `anomalib.metrics.per_image.pimo_numpy`."""

import numpy as np
import pytest
import torch
from numpy import ndarray
from torch import Tensor


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Generate tests for all functions in this module.

    All functions are parametrized with the same setting: 1 normal and 2 anomalous images.
    The anomaly maps are the same for all functions, but the masks are different.
    """
    expected_threshs = np.arange(1, 7 + 1, dtype=np.float32)
    shape = (1000, 1000)  # (H, W), 1 million pixels

    # --- normal ---
    # histogram of scores:
    # value:   7   6    5    4    3    2     1
    # count:   1   9   90  900   9k   90k  900k
    # cumsum:  1  10  100   1k  10k  100k    1M
    pred_norm = np.ones(1_000_000, dtype=np.float32)
    pred_norm[:100_000] += 1
    pred_norm[:10_000] += 1
    pred_norm[:1_000] += 1
    pred_norm[:100] += 1
    pred_norm[:10] += 1
    pred_norm[:1] += 1
    pred_norm = pred_norm.reshape(shape)
    mask_norm = np.zeros_like(pred_norm, dtype=np.int32)

    expected_fpr_norm = np.array([1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6], dtype=np.float64)
    expected_tpr_norm = np.full((7,), np.nan, dtype=np.float64)

    # --- anomalous ---
    pred_anom1 = pred_norm.copy()
    mask_anom1 = np.ones_like(pred_anom1, dtype=np.int32)
    expected_tpr_anom1 = expected_fpr_norm.copy()

    # only the first 100_000 pixels are anomalous
    # which corresponds to the first 100_000 highest scores (2 to 7)
    pred_anom2 = pred_norm.copy()
    mask_anom2 = np.concatenate([np.ones(100_000), np.zeros(900_000)]).reshape(shape).astype(np.int32)
    expected_tpr_anom2 = (10 * expected_fpr_norm).clip(0, 1)

    anomaly_maps = np.stack([pred_norm, pred_anom1, pred_anom2], axis=0)
    masks = np.stack([mask_norm, mask_anom1, mask_anom2], axis=0)

    expected_shared_fpr = expected_fpr_norm
    expected_per_image_tprs = np.stack([expected_tpr_norm, expected_tpr_anom1, expected_tpr_anom2], axis=0)
    expected_image_classes = np.array([0, 1, 1], dtype=np.int32)

    metafunc.parametrize(
        argnames=("binclf_algorithm",),
        argvalues=[("python",), ("numba",)],
    )

    if (
        metafunc.function is test_pimo_numpy
        or metafunc.function is test_pimo
        or metafunc.function is test_aupimo_values_numpy
        or metafunc.function is test_aupimo_values
    ):
        argvalues_arrays = [
            (
                anomaly_maps,
                masks,
                expected_threshs,
                expected_shared_fpr,
                expected_per_image_tprs,
                expected_image_classes,
            ),
            (
                10 * anomaly_maps,
                masks,
                10 * expected_threshs,
                expected_shared_fpr,
                expected_per_image_tprs,
                expected_image_classes,
            ),
        ]
        argvalues_tensors = [
            tuple(torch.from_numpy(arg) if isinstance(arg, ndarray) else arg for arg in arvals)
            for arvals in argvalues_arrays
        ]
        argvalues = argvalues_arrays if "numpy" in metafunc.function.__name__ else argvalues_tensors
        metafunc.parametrize(
            argnames=(
                "anomaly_maps",
                "masks",
                "expected_threshs",
                "expected_shared_fpr",
                "expected_per_image_tprs",
                "expected_image_classes",
            ),
            argvalues=argvalues,
        )

    if metafunc.function is test_aupimo_values_numpy or metafunc.function is test_aupimo_values:
        argvalues_arrays = [
            (
                (1e-1, 1.0),
                np.array(
                    [
                        np.nan,
                        # recall: trapezium area = (a + b) * h / 2
                        (0.10 + 1.0) * 1 / 2,
                        (1.0 + 1.0) * 1 / 2,
                    ],
                    dtype=np.float64,
                ),
            ),
            (
                (1e-3, 1e-1),
                np.array(
                    [
                        np.nan,
                        # average of two trapezium areas / 2 (normalizing factor)
                        (((1e-3 + 1e-2) * 1 / 2) + ((1e-2 + 1e-1) * 1 / 2)) / 2,
                        (((1e-2 + 1e-1) * 1 / 2) + ((1e-1 + 1.0) * 1 / 2)) / 2,
                    ],
                    dtype=np.float64,
                ),
            ),
            (
                (1e-5, 1e-4),
                np.array(
                    [
                        np.nan,
                        (1e-5 + 1e-4) * 1 / 2,
                        (1e-4 + 1e-3) * 1 / 2,
                    ],
                    dtype=np.float64,
                ),
            ),
        ]
        argvalues_tensors = [
            tuple(torch.from_numpy(arg) if isinstance(arg, ndarray) else arg for arg in arvals)
            for arvals in argvalues_arrays
        ]
        argvalues = argvalues_arrays if "numpy" in metafunc.function.__name__ else argvalues_tensors
        metafunc.parametrize(
            argnames=(
                "fpr_bounds",
                "expected_aupimos",  # trapezoid surfaces
            ),
            argvalues=argvalues,
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


def test_pimo_numpy(
    anomaly_maps: ndarray,
    masks: ndarray,
    binclf_algorithm: str,
    expected_threshs: ndarray,
    expected_shared_fpr: ndarray,
    expected_per_image_tprs: ndarray,
    expected_image_classes: ndarray,
) -> None:
    """Test if `pimo()` returns the expected values."""
    from anomalib.metrics.per_image import pimo_numpy

    threshs, shared_fpr, per_image_tprs, image_classes = pimo_numpy.pimo(
        anomaly_maps,
        masks,
        num_threshs=7,
        binclf_algorithm=binclf_algorithm,
        shared_fpr_metric="mean-per-image-fpr",
    )

    assert threshs.ndim == 1
    assert shared_fpr.ndim == 1
    assert per_image_tprs.ndim == 2
    assert image_classes.ndim == 1

    assert np.allclose(threshs, expected_threshs)
    assert np.allclose(shared_fpr, expected_shared_fpr)
    assert np.allclose(per_image_tprs, expected_per_image_tprs, equal_nan=True)
    assert (image_classes == expected_image_classes).all()


def test_pimo(
    anomaly_maps: Tensor,
    masks: Tensor,
    binclf_algorithm: str,
    expected_threshs: Tensor,
    expected_shared_fpr: Tensor,
    expected_per_image_tprs: Tensor,
    expected_image_classes: Tensor,
) -> None:
    """Test if `pimo()` returns the expected values."""
    from anomalib.metrics.per_image import pimo

    pimoresult = pimo.pimo(
        anomaly_maps,
        masks,
        num_threshs=7,
        binclf_algorithm=binclf_algorithm,
        shared_fpr_metric="mean-per-image-fpr",
    )
    threshs = pimoresult.threshs
    shared_fpr = pimoresult.shared_fpr
    per_image_tprs = pimoresult.per_image_tprs
    image_classes = pimoresult.image_classes

    # metadata
    assert pimoresult.shared_fpr_metric == "mean-per-image-fpr"
    # data
    assert threshs.ndim == 1
    assert shared_fpr.ndim == 1
    assert per_image_tprs.ndim == 2
    assert image_classes.ndim == 1

    assert torch.allclose(threshs, expected_threshs)
    assert torch.allclose(shared_fpr, expected_shared_fpr)
    assert torch.allclose(per_image_tprs, expected_per_image_tprs, equal_nan=True)
    assert (image_classes == expected_image_classes).all()


def test_aupimo_values_numpy(
    anomaly_maps: ndarray,
    masks: ndarray,
    binclf_algorithm: str,
    fpr_bounds: tuple[float, float],
    expected_threshs: ndarray,
    expected_shared_fpr: ndarray,
    expected_per_image_tprs: ndarray,
    expected_image_classes: ndarray,
    expected_aupimos: ndarray,
) -> None:
    """Test if `aupimo()` returns the expected values."""
    from anomalib.metrics.per_image import pimo_numpy

    threshs, shared_fpr, per_image_tprs, image_classes, aupimos = pimo_numpy.aupimo(
        anomaly_maps,
        masks,
        num_threshs=7,
        binclf_algorithm=binclf_algorithm,
        shared_fpr_metric="mean-per-image-fpr",
        fpr_bounds=fpr_bounds,
        force=True,
    )

    assert threshs.ndim == 1
    assert shared_fpr.ndim == 1
    assert per_image_tprs.ndim == 2
    assert image_classes.ndim == 1

    assert np.allclose(threshs, expected_threshs)
    assert np.allclose(shared_fpr, expected_shared_fpr)
    assert np.allclose(per_image_tprs, expected_per_image_tprs, equal_nan=True)
    assert (image_classes == expected_image_classes).all()

    assert aupimos.ndim == 1
    assert aupimos.shape == (3,)
    assert np.allclose(aupimos, expected_aupimos, equal_nan=True)


def test_aupimo_values(
    anomaly_maps: ndarray,
    masks: ndarray,
    binclf_algorithm: str,
    fpr_bounds: tuple[float, float],
    expected_threshs: ndarray,
    expected_shared_fpr: ndarray,
    expected_per_image_tprs: ndarray,
    expected_image_classes: ndarray,
    expected_aupimos: ndarray,
) -> None:
    """Test if `aupimo()` returns the expected values."""
    from anomalib.metrics.per_image import pimo

    pimoresult, aupimoresult = pimo.aupimo(
        anomaly_maps,
        masks,
        num_threshs=7,
        binclf_algorithm=binclf_algorithm,
        shared_fpr_metric="mean-per-image-fpr",
        fpr_bounds=fpr_bounds,
        force=True,
    )

    # from pimo result
    threshs = pimoresult.threshs
    shared_fpr = pimoresult.shared_fpr
    per_image_tprs = pimoresult.per_image_tprs
    image_classes = pimoresult.image_classes

    # from aupimo result
    fpr_lower_bound = aupimoresult.fpr_lower_bound
    fpr_upper_bound = aupimoresult.fpr_upper_bound
    thresh_lower_bound = aupimoresult.thresh_lower_bound
    thresh_upper_bound = aupimoresult.thresh_upper_bound
    num_threshs = aupimoresult.num_threshs
    aupimos = aupimoresult.aupimos

    # test metadata
    assert pimoresult.shared_fpr_metric == "mean-per-image-fpr"
    assert aupimoresult.shared_fpr_metric == "mean-per-image-fpr"
    assert (fpr_lower_bound, fpr_upper_bound) == fpr_bounds
    assert num_threshs == 7

    # test data
    assert threshs.ndim == 1
    assert shared_fpr.ndim == 1
    assert per_image_tprs.ndim == 2
    assert image_classes.ndim == 1
    assert anomaly_maps.min() <= thresh_lower_bound < thresh_upper_bound <= anomaly_maps.max()

    assert torch.allclose(threshs, expected_threshs)
    assert torch.allclose(shared_fpr, expected_shared_fpr)
    assert torch.allclose(per_image_tprs, expected_per_image_tprs, equal_nan=True)
    assert (image_classes == expected_image_classes).all()
    assert torch.allclose(aupimos, expected_aupimos, equal_nan=True)


def test_aupimo_edge(
    anomaly_maps: ndarray,
    masks: ndarray,
    binclf_algorithm: str,
    fpr_bounds: tuple[float, float],
) -> None:
    """Test some edge cases."""
    from anomalib.metrics.per_image import pimo_numpy

    # None is the case of testing the default bounds
    fpr_bounds = {"fpr_bounds": fpr_bounds, "shared_fpr_metric": "mean-per-image-fpr"} if fpr_bounds is not None else {}

    # not enough points on the curve
    # 10 threshs / 6 decades = 1.6 threshs per decade < 3
    with pytest.raises(RuntimeError):  # force=False --> raise error
        pimo_numpy.aupimo(
            anomaly_maps,
            masks,
            num_threshs=10,
            binclf_algorithm=binclf_algorithm,
            force=False,
            **fpr_bounds,
        )

    with pytest.warns(RuntimeWarning):  # force=True --> warn
        pimo_numpy.aupimo(
            anomaly_maps,
            masks,
            num_threshs=10,
            binclf_algorithm=binclf_algorithm,
            force=True,
            **fpr_bounds,
        )

    # default number of points on the curve (300k threshs) should be enough
    rng = np.random.default_rng(42)
    pimo_numpy.aupimo(
        anomaly_maps * rng.uniform(1.0, 1.1, size=anomaly_maps.shape),
        masks,
        # num_threshs=,
        binclf_algorithm=binclf_algorithm,
        force=False,
        **fpr_bounds,
    )
