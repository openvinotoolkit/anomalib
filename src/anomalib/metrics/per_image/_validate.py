"""Utils for validating arguments and results.

`torch` is imported in the functions that use it, so this module can be used in numpy-standalone mode.

author: jpcbertoldo
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy import ndarray


def is_tensor(tensor: Any, argname: str | None = None) -> None:  # noqa: ANN401
    """Validate that `tensor` is a `torch.Tensor`."""
    from torch import Tensor

    argname = f"'{argname}'" if argname is not None else "argument"

    if not isinstance(tensor, Tensor):
        msg = f"Expected {argname} to be a tensor, but got {type(tensor)}"
        raise TypeError(msg)


def num_threshs(num_threshs: int) -> None:
    """Validate the number of thresholds is a positive integer >= 2."""
    if not isinstance(num_threshs, int):
        msg = f"Expected the number of thresholds to be an integer, but got {type(num_threshs)}"
        raise TypeError(msg)

    if num_threshs < 2:
        msg = f"Expected the number of thresholds to be larger than 1, but got {num_threshs}"
        raise ValueError(msg)


def same_shape(*args) -> None:
    """Works both for tensors and ndarrays."""
    assert len(args) > 0
    shapes = sorted({tuple(arg.shape) for arg in args})
    if len(shapes) > 1:
        msg = f"Expected arguments to have the same shape, but got {shapes}"
        raise ValueError(msg)


def rate(rate: float | int, zero_ok: bool, one_ok: bool) -> None:
    """Validates a rate parameter.

    Args:
        rate (float | int): The rate to be validated.
        zero_ok (bool): Flag indicating if rate can be 0.
        one_ok (bool): Flag indicating if rate can be 1.
    """
    if not isinstance(rate, float | int):
        msg = f"Expected rate to be a float or int, but got {type(rate)}."
        raise TypeError(msg)

    if rate < 0.0 or rate > 1.0:
        msg = f"Expected rate to be in [0, 1], but got {rate}."
        raise ValueError(msg)

    if not zero_ok and rate == 0.0:
        msg = "Rate cannot be 0."
        raise ValueError(msg)

    if not one_ok and rate == 1.0:
        msg = "Rate cannot be 1."
        raise ValueError(msg)


def rate_range(bounds: tuple[float, float]) -> None:
    """Validates the range of rates within the bounds.

    Args:
        bounds (tuple[float, float]): The lower and upper bounds of the rates.
    """
    if not isinstance(bounds, tuple):
        msg = f"Expected the bounds to be a tuple, but got {type(bounds)}"
        raise TypeError(msg)

    if len(bounds) != 2:
        msg = f"Expected the bounds to be a tuple of length 2, but got {len(bounds)}"
        raise ValueError(msg)

    lower, upper = bounds
    rate(lower, zero_ok=False, one_ok=False)
    rate(upper, zero_ok=False, one_ok=True)

    if lower >= upper:
        msg = f"Expected the upper bound to be larger than the lower bound, but got {upper=} <= {lower=}"
        raise ValueError(msg)


def file_path(file_path: str | Path, must_exist: bool, extension: str | None, pathlib_ok: bool) -> None:
    """Validate the given path is a file (optionally) with the expected extension.

    Args:
        file_path (str | Path): The file path to validate.
        must_exist (bool): Flag indicating whether the file must exist.
        extension (str | None): The expected file extension, eg. .png, .jpg, etc. If `None`, no validation is performed.
        pathlib_ok (bool): Flag indicating whether `pathlib.Path` is allowed; if False, only `str` paths are allowed.
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    elif not isinstance(file_path, Path):
        msg = f"Expected file path to be a string or pathlib.Path, but got {type(file_path)}"
        raise TypeError(msg)

    # if it's here, then it's a `pathlib.Path`
    elif not pathlib_ok:
        msg = f"Only `str` paths are allowed, but got {type(file_path)}"
        raise TypeError(msg)

    if file_path.is_dir():
        msg = "Expected file path to be a file, but got a directory."
        raise ValueError(msg)

    if must_exist and not file_path.exists():
        msg = f"File does not exist: {file_path}"
        raise FileNotFoundError(msg)

    if extension is None:
        return

    if file_path.suffix != extension:
        msg = f"Expected file path to have extension '{extension}', but got '{file_path.suffix}'"
        raise ValueError(msg)


def file_paths(file_paths: list[str | Path], must_exist: bool, extension: str | None, pathlib_ok: bool) -> None:
    """Validate the given paths are files (optionally) with the expected extension.

    Args:
        file_paths (list[str | Path]): The file paths to validate.
        must_exist (bool): Flag indicating whether the files must exist.
        extension (str | None): The expected file extension, eg. .png, .jpg, etc. If `None`, no validation is performed.
        pathlib_ok (bool): Flag indicating whether `pathlib.Path` is allowed; if False, only `str` paths are allowed.
    """
    if not isinstance(file_paths, list):
        msg = f"Expected paths to be a list, but got {type(file_paths)}."
        raise TypeError(msg)

    for idx, path in enumerate(file_paths):
        try:
            msg = f"Invalid path at index {idx}: {path}"
            file_path(path, must_exist=must_exist, extension=extension, pathlib_ok=pathlib_ok)

        except TypeError as ex:  # noqa: PERF203
            raise TypeError(msg) from ex

        except ValueError as ex:
            raise ValueError(msg) from ex


def threshs(threshs: ndarray) -> None:
    """Validate that the thresholds are valid and monotonically increasing."""
    if not isinstance(threshs, ndarray):
        msg = f"Expected thresholds to be an ndarray, but got {type(threshs)}"
        raise TypeError(msg)

    if threshs.ndim != 1:
        msg = f"Expected thresholds to be 1D, but got {threshs.ndim}"
        raise ValueError(msg)

    if threshs.dtype.kind != "f":
        msg = f"Expected thresholds to be of float type, but got ndarray with dtype {threshs.dtype}"
        raise TypeError(msg)

    # make sure they are strictly increasing
    if not np.all(np.diff(threshs) > 0):
        msg = "Expected thresholds to be strictly increasing, but it is not."
        raise ValueError(msg)


def thresh_bounds(thresh_bounds: tuple[float, float]) -> None:
    if not isinstance(thresh_bounds, tuple):
        msg = f"Expected threshold bounds to be a tuple, but got {type(thresh_bounds)}."
        raise TypeError(msg)

    if len(thresh_bounds) != 2:
        msg = f"Expected threshold bounds to be a tuple of length 2, but got {len(thresh_bounds)}."
        raise ValueError(msg)

    lower, upper = thresh_bounds

    if not isinstance(lower, float):
        msg = f"Expected lower threshold bound to be a float, but got {type(lower)}."
        raise TypeError(msg)

    if not isinstance(upper, float):
        msg = f"Expected upper threshold bound to be a float, but got {type(upper)}."
        raise TypeError(msg)

    if upper <= lower:
        msg = f"Expected the upper bound to be greater than the lower bound, but got {upper} <= {lower}."
        raise ValueError(msg)


def anomaly_maps(anomaly_maps: ndarray) -> None:
    if not isinstance(anomaly_maps, ndarray):
        msg = f"Expected anomaly maps to be an ndarray, but got {type(anomaly_maps)}"
        raise TypeError(msg)

    if anomaly_maps.ndim != 3:
        msg = f"Expected anomaly maps have 3 dimensions (N, H, W), but got {anomaly_maps.ndim} dimensions"
        raise ValueError(msg)

    if anomaly_maps.dtype.kind != "f":
        msg = (
            "Expected anomaly maps to be an floating ndarray with anomaly scores,"
            f" but got ndarray with dtype {anomaly_maps.dtype}"
        )
        raise TypeError(msg)


def masks(masks: ndarray) -> None:
    if not isinstance(masks, ndarray):
        msg = f"Expected masks to be an ndarray, but got {type(masks)}"
        raise TypeError(msg)

    if masks.ndim != 3:
        msg = f"Expected masks have 3 dimensions (N, H, W), but got {masks.ndim} dimensions"
        raise ValueError(msg)

    if masks.dtype.kind == "b":
        pass

    elif masks.dtype.kind in ("i", "u"):
        masks_unique_vals = np.unique(masks)
        if np.any((masks_unique_vals != 0) & (masks_unique_vals != 1)):
            msg = (
                "Expected masks to be a *binary* ndarray with ground truth labels, "
                f"but got ndarray with unique values {sorted(masks_unique_vals)}"
            )
            raise ValueError(msg)

    else:
        msg = (
            "Expected masks to be an integer or boolean ndarray with ground truth labels, "
            f"but got ndarray with dtype {masks.dtype}"
        )
        raise TypeError(msg)


def binclf_curves(binclf_curves: ndarray, valid_threshs: ndarray | None) -> None:
    if not isinstance(binclf_curves, ndarray):
        msg = f"Expected binclf curves to be an ndarray, but got {type(binclf_curves)}"
        raise TypeError(msg)

    if binclf_curves.ndim != 4:
        msg = f"Expected binclf curves to be 4D, but got {binclf_curves.ndim}D"
        raise ValueError(msg)

    if binclf_curves.shape[-2:] != (2, 2):
        msg = f"Expected binclf curves to have shape (..., 2, 2), but got {binclf_curves.shape}"
        raise ValueError(msg)

    if binclf_curves.dtype != np.int64:
        msg = f"Expected binclf curves to have dtype int64, but got {binclf_curves.dtype}."
        raise TypeError(msg)

    if (binclf_curves < 0).any():
        msg = "Expected binclf curves to have non-negative values, but got negative values."
        raise ValueError(msg)

    neg = binclf_curves[:, :, 0, :].sum(axis=-1)  # (num_images, num_threshs)

    if (neg != neg[:, :1]).any():
        msg = "Expected binclf curves to have the same number of negatives per image for every thresh."
        raise ValueError(msg)

    pos = binclf_curves[:, :, 1, :].sum(axis=-1)  # (num_images, num_threshs)

    if (pos != pos[:, :1]).any():
        msg = "Expected binclf curves to have the same number of positives per image for every thresh."
        raise ValueError(msg)

    if valid_threshs is None:
        return

    if binclf_curves.shape[1] != valid_threshs.shape[0]:
        msg = (
            "Expected the binclf curves to have as many confusion matrices as the thresholds sequence, "
            f"but got {binclf_curves.shape[1]} and {valid_threshs.shape[0]}"
        )
        raise RuntimeError(msg)


def images_classes(images_classes: ndarray) -> None:
    if not isinstance(images_classes, ndarray):
        msg = f"Expected image classes to be an ndarray, but got {type(images_classes)}."
        raise TypeError(msg)

    if images_classes.ndim != 1:
        msg = f"Expected image classes to be 1D, but got {images_classes.ndim}D."
        raise ValueError(msg)

    if images_classes.dtype.kind == "b":
        pass
    elif images_classes.dtype.kind in ("i", "u"):
        unique_vals = np.unique(images_classes)
        if np.any((unique_vals != 0) & (unique_vals != 1)):
            msg = (
                "Expected image classes to be a *binary* ndarray with ground truth labels, "
                f"but got ndarray with unique values {sorted(unique_vals)}"
            )
            raise ValueError(msg)
    else:
        msg = (
            "Expected image classes to be an integer or boolean ndarray with ground truth labels, "
            f"but got ndarray with dtype {images_classes.dtype}"
        )
        raise TypeError(msg)


def rates(rates: ndarray, nan_allowed: bool) -> None:
    if not isinstance(rates, ndarray):
        msg = f"Expected rates to be an ndarray, but got {type(rates)}."
        raise TypeError(msg)

    if rates.ndim != 1:
        msg = f"Expected rates to be 1D, but got {rates.ndim}D."
        raise ValueError(msg)

    if rates.dtype.kind != "f":
        msg = f"Expected rates to have dtype of float type, but got {rates.dtype}."
        raise ValueError(msg)

    isnan_mask = np.isnan(rates)
    if nan_allowed:
        # if they are all nan, then there is nothing to validate
        if isnan_mask.all():
            return
        valid_values = rates[~isnan_mask]
    elif isnan_mask.any():
        msg = "Expected rates to not contain NaN values, but got NaN values."
        raise ValueError(msg)
    else:
        valid_values = rates

    if (valid_values < 0).any():
        msg = "Expected rates to have values in the interval [0, 1], but got values < 0."
        raise ValueError(msg)

    if (valid_values > 1).any():
        msg = "Expected rates to have values in the interval [0, 1], but got values > 1."
        raise ValueError(msg)


def rate_curve(rate_curve: ndarray, nan_allowed: bool, decreasing: bool) -> None:
    rates(rate_curve, nan_allowed=nan_allowed)

    diffs = np.diff(rate_curve)
    diffs_valid = diffs[~np.isnan(diffs)] if nan_allowed else diffs

    if decreasing and (diffs_valid > 0).any():
        msg = "Expected rate curve to be monotonically decreasing, but got non-monotonically decreasing values."
        raise ValueError(msg)

    if not decreasing and (diffs_valid < 0).any():
        msg = "Expected rate curve to be monotonically increasing, but got non-monotonically increasing values."
        raise ValueError(msg)


def per_image_rate_curves(rate_curves: ndarray, nan_allowed: bool, decreasing: bool | None) -> None:
    if not isinstance(rate_curves, ndarray):
        msg = f"Expected per-image rate curves to be an ndarray, but got {type(rate_curves)}."
        raise TypeError(msg)

    if rate_curves.ndim != 2:
        msg = f"Expected per-image rate curves to be 2D, but got {rate_curves.ndim}D."
        raise ValueError(msg)

    if rate_curves.dtype.kind != "f":
        msg = f"Expected per-image rate curves to have dtype of float type, but got {rate_curves.dtype}."
        raise ValueError(msg)

    isnan_mask = np.isnan(rate_curves)
    if nan_allowed:
        # if they are all nan, then there is nothing to validate
        if isnan_mask.all():
            return
        valid_values = rate_curves[~isnan_mask]
    elif isnan_mask.any():
        msg = "Expected per-image rate curves to not contain NaN values, but got NaN values."
        raise ValueError(msg)
    else:
        valid_values = rate_curves

    if (valid_values < 0).any():
        msg = "Expected per-image rate curves to have values in the interval [0, 1], but got values < 0."
        raise ValueError(msg)

    if (valid_values > 1).any():
        msg = "Expected per-image rate curves to have values in the interval [0, 1], but got values > 1."
        raise ValueError(msg)

    if decreasing is None:
        return

    diffs = np.diff(rate_curves, axis=1)
    diffs_valid = diffs[~np.isnan(diffs)] if nan_allowed else diffs

    if decreasing and (diffs_valid > 0).any():
        msg = (
            "Expected per-image rate curves to be monotonically decreasing, "
            "but got non-monotonically decreasing values."
        )
        raise ValueError(msg)

    if not decreasing and (diffs_valid < 0).any():
        msg = (
            "Expected per-image rate curves to be monotonically increasing, "
            "but got non-monotonically increasing values."
        )
        raise ValueError(msg)
