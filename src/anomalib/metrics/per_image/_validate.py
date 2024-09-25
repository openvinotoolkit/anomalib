"""Utils for validating arguments and results.

TODO(jpcbertoldo): Move validations to a common place and reuse them across the codebase.
https://github.com/openvinotoolkit/anomalib/issues/2093
"""

# Original Code
# https://github.com/jpcbertoldo/aupimo
#
# Modified
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .utils import images_classes_from_masks

if TYPE_CHECKING:
    from .pimo import AUPIMOResult

logger = logging.getLogger(__name__)


def is_num_threshs_gte2(num_threshs: int) -> None:
    """Validate the number of thresholds is a positive integer >= 2."""
    if not isinstance(num_threshs, int):
        msg = f"Expected the number of thresholds to be an integer, but got {type(num_threshs)}"
        raise TypeError(msg)

    if num_threshs < 2:
        msg = f"Expected the number of thresholds to be larger than 1, but got {num_threshs}"
        raise ValueError(msg)


def is_same_shape(*args) -> None:
    """Works both for tensors and ndarrays."""
    assert len(args) > 0
    shapes = sorted({tuple(arg.shape) for arg in args})
    if len(shapes) > 1:
        msg = f"Expected arguments to have the same shape, but got {shapes}"
        raise ValueError(msg)


def is_rate(rate: float | int, zero_ok: bool, one_ok: bool) -> None:
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


def is_rate_range(bounds: tuple[float, float]) -> None:
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
    is_rate(lower, zero_ok=False, one_ok=False)
    is_rate(upper, zero_ok=False, one_ok=True)

    if lower >= upper:
        msg = f"Expected the upper bound to be larger than the lower bound, but got {upper=} <= {lower=}"
        raise ValueError(msg)


def is_threshs(threshs: Tensor) -> None:
    """Validate that the thresholds are valid and monotonically increasing."""
    if not isinstance(threshs, Tensor):
        msg = f"Expected thresholds to be an Tensor, but got {type(threshs)}"
        raise TypeError(msg)

    if threshs.ndim != 1:
        msg = f"Expected thresholds to be 1D, but got {threshs.ndim}"
        raise ValueError(msg)

    if not threshs.dtype.is_floating_point:
        msg = f"Expected thresholds to be of float type, but got Tensor with dtype {threshs.dtype}"
        raise TypeError(msg)

    # make sure they are strictly increasing
    if not torch.all(torch.diff(threshs) > 0):
        msg = "Expected thresholds to be strictly increasing, but it is not."
        raise ValueError(msg)


def is_thresh_bounds(thresh_bounds: tuple[float, float]) -> None:
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


def is_anomaly_maps(anomaly_maps: Tensor) -> None:
    if not isinstance(anomaly_maps, Tensor):
        msg = f"Expected anomaly maps to be an Tensor, but got {type(anomaly_maps)}"
        raise TypeError(msg)

    if anomaly_maps.ndim != 3:
        msg = f"Expected anomaly maps have 3 dimensions (N, H, W), but got {anomaly_maps.ndim} dimensions"
        raise ValueError(msg)

    if not anomaly_maps.dtype.is_floating_point:
        msg = (
            "Expected anomaly maps to be an floating Tensor with anomaly scores,"
            f" but got Tensor with dtype {anomaly_maps.dtype}"
        )
        raise TypeError(msg)


def is_masks(masks: Tensor) -> None:
    if not isinstance(masks, Tensor):
        msg = f"Expected masks to be an Tensor, but got {type(masks)}"
        raise TypeError(msg)

    if masks.ndim != 3:
        msg = f"Expected masks have 3 dimensions (N, H, W), but got {masks.ndim} dimensions"
        raise ValueError(msg)

    if masks.dtype == torch.bool:
        pass
    elif masks.dtype.is_floating_point:
        msg = (
            "Expected masks to be an integer or boolean Tensor with ground truth labels, "
            f"but got Tensor with dtype {masks.dtype}"
        )
        raise TypeError(msg)
    else:
        # assumes the type to be (signed or unsigned) integer
        # this will change with the dataclass refactor
        masks_unique_vals = torch.unique(masks)
        if torch.any((masks_unique_vals != 0) & (masks_unique_vals != 1)):
            msg = (
                "Expected masks to be a *binary* Tensor with ground truth labels, "
                f"but got Tensor with unique values {sorted(masks_unique_vals)}"
            )
            raise ValueError(msg)


def is_binclf_curves(binclf_curves: Tensor, valid_threshs: Tensor | None) -> None:
    if not isinstance(binclf_curves, Tensor):
        msg = f"Expected binclf curves to be an Tensor, but got {type(binclf_curves)}"
        raise TypeError(msg)

    if binclf_curves.ndim != 4:
        msg = f"Expected binclf curves to be 4D, but got {binclf_curves.ndim}D"
        raise ValueError(msg)

    if binclf_curves.shape[-2:] != (2, 2):
        msg = f"Expected binclf curves to have shape (..., 2, 2), but got {binclf_curves.shape}"
        raise ValueError(msg)

    if binclf_curves.dtype != torch.int64:
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


def is_images_classes(images_classes: Tensor) -> None:
    if not isinstance(images_classes, Tensor):
        msg = f"Expected image classes to be an Tensor, but got {type(images_classes)}."
        raise TypeError(msg)

    if images_classes.ndim != 1:
        msg = f"Expected image classes to be 1D, but got {images_classes.ndim}D."
        raise ValueError(msg)

    if images_classes.dtype == torch.bool:
        pass
    elif images_classes.dtype.is_floating_point:
        msg = (
            "Expected image classes to be an integer or boolean Tensor with ground truth labels, "
            f"but got Tensor with dtype {images_classes.dtype}"
        )
        raise TypeError(msg)
    else:
        # assumes the type to be (signed or unsigned) integer
        # this will change with the dataclass refactor
        unique_vals = torch.unique(images_classes)
        if torch.any((unique_vals != 0) & (unique_vals != 1)):
            msg = (
                "Expected image classes to be a *binary* Tensor with ground truth labels, "
                f"but got Tensor with unique values {sorted(unique_vals)}"
            )
            raise ValueError(msg)


def is_rates(rates: Tensor, nan_allowed: bool) -> None:
    if not isinstance(rates, Tensor):
        msg = f"Expected rates to be an Tensor, but got {type(rates)}."
        raise TypeError(msg)

    if rates.ndim != 1:
        msg = f"Expected rates to be 1D, but got {rates.ndim}D."
        raise ValueError(msg)

    if not rates.dtype.is_floating_point:
        msg = f"Expected rates to have dtype of float type, but got {rates.dtype}."
        raise ValueError(msg)

    isnan_mask = torch.isnan(rates)
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


def is_rate_curve(rate_curve: Tensor, nan_allowed: bool, decreasing: bool) -> None:
    is_rates(rate_curve, nan_allowed=nan_allowed)

    diffs = torch.diff(rate_curve)
    diffs_valid = diffs[~torch.isnan(diffs)] if nan_allowed else diffs

    if decreasing and (diffs_valid > 0).any():
        msg = "Expected rate curve to be monotonically decreasing, but got non-monotonically decreasing values."
        raise ValueError(msg)

    if not decreasing and (diffs_valid < 0).any():
        msg = "Expected rate curve to be monotonically increasing, but got non-monotonically increasing values."
        raise ValueError(msg)


def is_per_image_rate_curves(rate_curves: Tensor, nan_allowed: bool, decreasing: bool | None) -> None:
    if not isinstance(rate_curves, Tensor):
        msg = f"Expected per-image rate curves to be an Tensor, but got {type(rate_curves)}."
        raise TypeError(msg)

    if rate_curves.ndim != 2:
        msg = f"Expected per-image rate curves to be 2D, but got {rate_curves.ndim}D."
        raise ValueError(msg)

    if not rate_curves.dtype.is_floating_point:
        msg = f"Expected per-image rate curves to have dtype of float type, but got {rate_curves.dtype}."
        raise ValueError(msg)

    isnan_mask = torch.isnan(rate_curves)
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

    diffs = torch.diff(rate_curves, axis=1)
    diffs_valid = diffs[~torch.isnan(diffs)] if nan_allowed else diffs

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


def is_scores_batch(scores_batch: torch.Tensor) -> None:
    """scores_batch (torch.Tensor): floating (N, D)."""
    if not isinstance(scores_batch, torch.Tensor):
        msg = f"Expected `scores_batch` to be an torch.Tensor, but got {type(scores_batch)}"
        raise TypeError(msg)

    if not scores_batch.dtype.is_floating_point:
        msg = (
            "Expected `scores_batch` to be an floating torch.Tensor with anomaly scores_batch,"
            f" but got torch.Tensor with dtype {scores_batch.dtype}"
        )
        raise TypeError(msg)

    if scores_batch.ndim != 2:
        msg = f"Expected `scores_batch` to be 2D, but got {scores_batch.ndim}"
        raise ValueError(msg)


def is_gts_batch(gts_batch: torch.Tensor) -> None:
    """gts_batch (torch.Tensor): boolean (N, D)."""
    if not isinstance(gts_batch, torch.Tensor):
        msg = f"Expected `gts_batch` to be an torch.Tensor, but got {type(gts_batch)}"
        raise TypeError(msg)

    if gts_batch.dtype != torch.bool:
        msg = (
            "Expected `gts_batch` to be an boolean torch.Tensor with anomaly scores_batch,"
            f" but got torch.Tensor with dtype {gts_batch.dtype}"
        )
        raise TypeError(msg)

    if gts_batch.ndim != 2:
        msg = f"Expected `gts_batch` to be 2D, but got {gts_batch.ndim}"
        raise ValueError(msg)


def has_at_least_one_anomalous_image(masks: torch.Tensor) -> None:
    is_masks(masks)
    image_classes = images_classes_from_masks(masks)
    if (image_classes == 1).sum() == 0:
        msg = "Expected at least one ANOMALOUS image, but found none."
        raise ValueError(msg)


def has_at_least_one_normal_image(masks: torch.Tensor) -> None:
    is_masks(masks)
    image_classes = images_classes_from_masks(masks)
    if (image_classes == 0).sum() == 0:
        msg = "Expected at least one NORMAL image, but found none."
        raise ValueError(msg)


def joint_validate_threshs_shared_fpr(threshs: torch.Tensor, shared_fpr: torch.Tensor) -> None:
    if threshs.shape[0] != shared_fpr.shape[0]:
        msg = (
            "Expected `threshs` and `shared_fpr` to have the same number of elements, "
            f"but got {threshs.shape[0]} != {shared_fpr.shape[0]}"
        )
        raise ValueError(msg)


def is_per_image_tprs(per_image_tprs: torch.Tensor, image_classes: torch.Tensor) -> None:
    is_images_classes(image_classes)
    # general validations
    is_per_image_rate_curves(
        per_image_tprs,
        nan_allowed=True,  # normal images have NaN TPRs
        decreasing=None,  # not checked here
    )

    # specific to anomalous images
    is_per_image_rate_curves(
        per_image_tprs[image_classes == 1],
        nan_allowed=False,
        decreasing=True,
    )

    # specific to normal images
    normal_images_tprs = per_image_tprs[image_classes == 0]
    if not normal_images_tprs.isnan().all():
        msg = "Expected all normal images to have NaN TPRs, but some have non-NaN values."
        raise ValueError(msg)


def is_per_image_scores(per_image_scores: torch.Tensor) -> None:
    if per_image_scores.ndim != 1:
        msg = f"Expected per-image scores to be 1D, but got {per_image_scores.ndim}D."
        raise ValueError(msg)


def is_image_class(image_class: int) -> None:
    if image_class not in {0, 1}:
        msg = f"Expected image class to be either 0 for 'normal' or 1 for 'anomalous', but got {image_class}."
        raise ValueError(msg)


def is_models_ordered(models_ordered: tuple[str, ...]) -> None:
    if not isinstance(models_ordered, tuple):
        msg = f"Expected models ordered to be a tuple, but got {type(models_ordered)}."
        raise TypeError(msg)

    if len(models_ordered) < 2:
        msg = f"Expected models ordered to have at least 2 models, but got {len(models_ordered)}."
        raise ValueError(msg)

    for model_name in models_ordered:
        if not isinstance(model_name, str):
            msg = f"Expected model name to be a string, but got {type(model_name)} for model {model_name}."
            raise TypeError(msg)

        if model_name == "":
            msg = "Expected model name to be non-empty, but got empty string."
            raise ValueError(msg)

    num_redundant_models = len(models_ordered) - len(set(models_ordered))
    if num_redundant_models > 0:
        msg = f"Expected models ordered to have unique models, but got {num_redundant_models} redundant models."
        raise ValueError(msg)


def is_confidences(confidences: dict[tuple[str, str], float]) -> None:
    if not isinstance(confidences, dict):
        msg = f"Expected confidences to be a dict, but got {type(confidences)}."
        raise TypeError(msg)

    for (model1, model2), confidence in confidences.items():
        if not isinstance(model1, str):
            msg = f"Expected model name to be a string, but got {type(model1)} for model {model1}."
            raise TypeError(msg)

        if not isinstance(model2, str):
            msg = f"Expected model name to be a string, but got {type(model2)} for model {model2}."
            raise TypeError(msg)

        if not isinstance(confidence, float):
            msg = f"Expected confidence to be a float, but got {type(confidence)} for models {model1} and {model2}."
            raise TypeError(msg)

        if not (0 <= confidence <= 1):
            msg = f"Expected confidence to be between 0 and 1, but got {confidence} for models {model1} and {model2}."
            raise ValueError(msg)


def joint_validate_models_ordered_and_confidences(
    models_ordered: tuple[str, ...],
    confidences: dict[tuple[str, str], float],
) -> None:
    num_models = len(models_ordered)
    expected_num_pairs = num_models * (num_models - 1)

    if len(confidences) != expected_num_pairs:
        msg = f"Expected {expected_num_pairs} pairs of models, but got {len(confidences)} pairs of models."
        raise ValueError(msg)

    models_in_confidences = {model for pair_models in confidences for model in pair_models}

    diff = set(models_ordered).symmetric_difference(models_in_confidences)
    if len(diff) > 0:
        msg = (
            "Expected models in confidences to be the same as models ordered, but got models missing in one"
            f"of them: {diff}."
        )
        raise ValueError(msg)


def is_scores_per_model_tensor(scores_per_model: dict[str, Tensor] | OrderedDict[str, Tensor]) -> None:
    first_key_value = None

    for model_name, scores in scores_per_model.items():
        if scores.ndim != 1:
            msg = f"Expected scores to be 1D, but got {scores.ndim}D for model {model_name}."
            raise ValueError(msg)

        num_valid_scores = scores[~torch.isnan(scores)].numel()

        if num_valid_scores < 1:
            msg = f"Expected at least 1 non-nan score, but got {num_valid_scores} for model {model_name}."
            raise ValueError(msg)

        if first_key_value is None:
            first_key_value = (model_name, scores)
            continue

        first_model_name, first_scores = first_key_value

        # same shape
        if scores.shape[0] != first_scores.shape[0]:
            msg = (
                "Expected scores to have the same number of scores, "
                f"but got ({model_name}) {scores.shape[0]} != {first_scores.shape[0]} ({first_model_name})."
            )
            raise ValueError(msg)

        # `nan` at the same indices
        if (torch.isnan(scores) != torch.isnan(first_scores)).any():
            msg = (
                "Expected `nan` values, if any, to be at the same indices, "
                f"but there are differences between models {model_name} and {first_model_name}."
            )
            raise ValueError(msg)


def is_scores_per_model_aupimoresult(
    scores_per_model: dict[str, "AUPIMOResult"] | OrderedDict[str, "AUPIMOResult"],
) -> None:
    first_key_value = None

    for model_name, aupimoresult in scores_per_model.items():
        if first_key_value is None:
            first_key_value = (model_name, aupimoresult)
            continue

        first_model_name, first_aupimoresult = first_key_value

        if aupimoresult.fpr_bounds != first_aupimoresult.fpr_bounds:
            msg = (
                "Expected AUPIMOResult objects in scores per model to have the same FPR bounds, "
                f"but got ({model_name}) {aupimoresult.fpr_bounds} != "
                f"{first_aupimoresult.fpr_bounds} ({first_model_name})."
            )
            raise ValueError(msg)


def is_scores_per_model(
    scores_per_model: dict[str, Tensor]
    | OrderedDict[str, Tensor]
    | dict[str, "AUPIMOResult"]
    | OrderedDict[str, "AUPIMOResult"],
) -> None:
    # it has to be imported here to avoid circular imports
    from .pimo import AUPIMOResult

    if not isinstance(scores_per_model, dict | OrderedDict):
        msg = f"Expected scores per model to be a dictionary or ordered dictionary, but got {type(scores_per_model)}."
        raise TypeError(msg)

    if len(scores_per_model) < 2:
        msg = f"Expected scores per model to have at least 2 models, but got {len(scores_per_model)}."
        raise ValueError(msg)

    if not all(isinstance(model_name, str) for model_name in scores_per_model):
        msg = "Expected scores per model to have model names (strings) as keys."
        raise TypeError(msg)

    first_instance = next(iter(scores_per_model.values()))

    if (
        isinstance(first_instance, Tensor)
        and any(not isinstance(scores, Tensor) for scores in scores_per_model.values())
    ) or (
        isinstance(first_instance, AUPIMOResult)
        and any(not isinstance(scores, AUPIMOResult) for scores in scores_per_model.values())
    ):
        msg = (
            "Values in the scores per model dict must have the same type for values (Tensor or AUPIMOResult), "
            "but more than one type was found."
        )
        raise TypeError(msg)

    if isinstance(first_instance, Tensor):
        is_scores_per_model_tensor(scores_per_model)
        return

    is_scores_per_model_tensor(
        {model_name: scores.aupimos for model_name, scores in scores_per_model.items()},
    )

    is_scores_per_model_aupimoresult(scores_per_model)
