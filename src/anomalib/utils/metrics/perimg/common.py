from __future__ import annotations

import torch
from torch import Tensor

# =========================================== ARGS VALIDATION ===========================================


def _validate_rate_curve(curve: Tensor, nan_allowed: bool = False):
    if not isinstance(curve, Tensor):
        raise ValueError(f"Expected argument `curve` to be a Tensor, but got {type(curve)}.")

    if curve.ndim != 1:
        raise ValueError(f"Expected argument `curve` to be a 1D tensor, but got {curve.ndim}D tensor.")

    if not torch.is_floating_point(curve):
        raise ValueError(f"Expected argument `curve` to have dtype float, but got {curve.dtype}.")

    if not nan_allowed:
        if torch.isnan(curve).any():
            raise ValueError("Expected argument `curve` to not contain NaN values, but got NaN values.")
        valid_values = curve
    else:
        valid_values = curve[~torch.isnan(curve)]

    if (valid_values < 0).any() or (valid_values > 1).any():
        raise ValueError(
            "Expected argument `curve` to have values in the interval [0, 1], but got values outside this interval."
        )

    diffs = curve.diff()
    diffs_valid = diffs if not nan_allowed else diffs[~torch.isnan(diffs)]

    if (diffs_valid > 0).any():
        raise ValueError(
            "Expected argument `curve` to be monotonically decreasing, but got non-monotonically decreasing values."
        )


def _validate_perimg_rate_curves(curves: Tensor, nan_allowed: bool = False):
    if not isinstance(curves, Tensor):
        raise ValueError(f"Expected argument `curves` to be a Tensor, but got {type(curves)}.")

    if curves.ndim != 2:
        raise ValueError(f"Expected argument `curves` to be a 2D tensor, but got {curves.ndim}D tensor.")

    if not torch.is_floating_point(curves):
        raise ValueError(f"Expected argument `curves` to have dtype float, but got {curves.dtype}.")

    if not nan_allowed:
        if torch.isnan(curves).any():
            raise ValueError("Expected argument `curves` to not contain NaN values, but got NaN values.")
        valid_values = curves
    else:
        valid_values = curves[~torch.isnan(curves)]

    if (valid_values < 0).any() or (valid_values > 1).any():
        raise ValueError(
            "Expected argument `curves` to have values in the interval [0, 1], but got values outside this interval."
        )

    diffs = curves.diff(dim=1)
    diffs_valid = diffs if not nan_allowed else diffs[~torch.isnan(diffs)]

    if (diffs_valid > 0).any():
        raise ValueError(
            "Expected argument `curves` to be monotonically decreasing, but got non-monotonically decreasing values."
        )


def _validate_thresholds(thresholds: Tensor):
    if not isinstance(thresholds, Tensor):
        raise ValueError(f"Expected argument `thresholds` to be a Tensor, but got {type(thresholds)}.")

    if thresholds.ndim != 1:
        raise ValueError(f"Expected argument `thresholds` to be a 1D tensor, but got {thresholds.ndim}D tensor.")

    if not torch.is_floating_point(thresholds):
        raise ValueError(f"Expected argument `thresholds` to have dtype float, but got {thresholds.dtype}.")

    diffs = thresholds.diff()
    if (diffs <= 0).any():
        raise ValueError("Expected argument `thresholds` to be strictly increasing (thresholds[k+1] > thresholds[k]), ")


def _validate_image_classes(image_classes: Tensor):
    if not isinstance(image_classes, Tensor):
        raise ValueError(f"Expected argument `image_classes` to be a Tensor, but got {type(image_classes)}.")

    if image_classes.ndim != 1:
        raise ValueError(f"Expected argument `image_classes` to be a 1D tensor, but got {image_classes.ndim}D tensor.")

    if torch.is_floating_point(image_classes):
        raise ValueError(
            "Expected argument `image_classes` to be an int or long tensor with ground truth labels, "
            f"but got a float tensor with values {image_classes.dtype}."
        )

    unique_values = torch.unique(image_classes)
    if torch.any((unique_values != 0) & (unique_values != 1)):
        raise ValueError(
            "Expected argument `image_classes` to be a *binary* tensor with ground truth labels, "
            f"but got a tensor with values {unique_values}."
        )


def _validate_atleast_one_anomalous_image(image_classes: Tensor):
    if (image_classes == 1).sum() == 0:
        raise ValueError("Expected argument at least one anomalous image, but found none.")


def _validate_atleast_one_normal_image(image_classes: Tensor):
    if (image_classes == 0).sum() == 0:
        raise ValueError("Expected argument at least one normal image, but found none.")
