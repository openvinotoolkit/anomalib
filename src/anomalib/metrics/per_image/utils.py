"""Torch-oriented interfaces for `utils.py`."""
from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from . import _validate, utils_numpy
from .utils_numpy import StatsOutliersPolicy, StatsRepeatedPolicy

if TYPE_CHECKING:
    from .pimo import AUPIMOResult


def per_image_scores_stats(
    per_image_scores: Tensor,
    images_classes: Tensor | None = None,
    only_class: int | None = None,
    outliers_policy: str | None = StatsOutliersPolicy.NONE,
    repeated_policy: str | None = StatsRepeatedPolicy.AVOID,
    repeated_replacement_atol: float = 1e-2,
) -> list[dict[str, str | int | float]]:
    """Torch-oriented interface for `per_image_scores_stats`. See its dscription for more details (below).

    Numpy version docstring
    =======================

    {docstring}
    """
    _validate.is_tensor(per_image_scores, "per_image_scores")
    per_image_scores_array = per_image_scores.detach().cpu().numpy()

    if images_classes is not None:
        _validate.is_tensor(images_classes, "images_classes")
        images_classes_array = images_classes.detach().cpu().numpy()

    else:
        images_classes_array = None

    # other validations happen inside `utils_numpy.per_image_scores_stats`

    return utils_numpy.per_image_scores_stats(
        per_image_scores_array,
        images_classes_array,
        only_class=only_class,
        outliers_policy=outliers_policy,
        repeated_policy=repeated_policy,
        repeated_replacement_atol=repeated_replacement_atol,
    )


per_image_scores_stats.__doc__ = per_image_scores_stats.__doc__.format(  # type: ignore[union-attr]
    docstring=utils_numpy.per_image_scores_stats.__doc__,
)


def _validate_scores_per_model(  # noqa: C901
    scores_per_model: dict[str, Tensor]
    | OrderedDict[str, Tensor]
    | dict[str, AUPIMOResult]
    | OrderedDict[str, AUPIMOResult],
) -> None:
    # it has to be imported here to avoid circular imports
    from .pimo import AUPIMOResult

    if not isinstance(scores_per_model, dict | OrderedDict):
        msg = f"Expected scores per model to be a dictionary or ordered dictionary, but got {type(scores_per_model)}."
        raise TypeError(msg)

    if len(scores_per_model) < 2:
        msg = f"Expected scores per model to have at least 2 models, but got {len(scores_per_model)}."
        raise ValueError(msg)

    first_key_value_tensor = None

    for model_name, scores in scores_per_model.items():
        if not isinstance(model_name, str):
            msg = f"Expected model name to be a string, but got {type(model_name)} for model {model_name}."
            raise TypeError(msg)

        if isinstance(scores, AUPIMOResult):
            scores_tensor = scores.aupimos
        elif isinstance(scores, Tensor):
            scores_tensor = scores
        else:
            msg = f"Expected scores to be a Tensor or AUPIMOResult, but got {type(scores)} for model {model_name}."
            raise TypeError(msg)

        if scores_tensor.ndim != 1:
            msg = f"Expected scores to be 1D Tensor, but got {scores_tensor.ndim}D for model {model_name}."
            raise ValueError(msg)

        num_valid_scores = scores_tensor[~torch.isnan(scores_tensor)].numel()

        if num_valid_scores < 2:
            msg = f"Expected at least 2 scores, but got {num_valid_scores} for model {model_name}."
            raise ValueError(msg)

        if first_key_value_tensor is None:
            first_key_value_tensor = (model_name, scores, scores_tensor)
            continue

        first_model_name, first_scores, first_scores_tensor = first_key_value_tensor

        # must have the same type
        # test using `isinstance` to avoid issues with subclasses
        if isinstance(scores, Tensor) != isinstance(first_scores, Tensor):
            msg = (
                "Expected scores to have the same type, "
                f"but got ({model_name}) {type(scores)} != {type(first_scores)} ({first_model_name})."
            )
            raise TypeError(msg)

        # same shape
        if scores_tensor.shape != first_scores_tensor.shape:
            msg = (
                "Expected scores to have the same shape, "
                f"but got ({model_name}) {scores_tensor.shape} != {first_scores_tensor.shape} ({first_model_name})."
            )
            raise ValueError(msg)

        # `nan` at the same indices
        if (torch.isnan(scores_tensor) != torch.isnan(first_scores_tensor)).any():
            msg = (
                "Expected `nan` values, if any, to be at the same indices, "
                f"but there are differences between models {model_name} and {first_model_name}."
            )
            raise ValueError(msg)

        if isinstance(scores, Tensor):
            continue

        # check that the metadata is the same, so they can be compared indeed

        if scores.shared_fpr_metric != first_scores.shared_fpr_metric:
            msg = (
                "Expected scores to have the same shared FPR metric, "
                f"but got ({model_name}) {scores.shared_fpr_metric} != "
                f"{first_scores.shared_fpr_metric} ({first_model_name})."
            )
            raise ValueError(msg)

        if scores.fpr_bounds != first_scores.fpr_bounds:
            msg = (
                "Expected scores to have the same FPR bounds, "
                f"but got ({model_name}) {scores.fpr_bounds} != {first_scores.fpr_bounds} ({first_model_name})."
            )
            raise ValueError(msg)


def compare_models_pairwise_ttest(
    scores_per_model: dict[str, Tensor]
    | OrderedDict[str, Tensor]
    | dict[str, AUPIMOResult]
    | OrderedDict[str, AUPIMOResult],
    alternative: str,
    higher_is_better: bool,
) -> tuple[tuple[str, ...], dict[tuple[str, str], float]]:
    """Torch-oriented interface for `compare_models_pairwise_ttest`. See its dscription for more details (below).

    Numpy version docstring
    =======================

    {docstring}
    """
    _validate_scores_per_model(scores_per_model)
    scores_per_model_items = [
        (
            model_name,
            (scores if isinstance(scores, Tensor) else scores.aupimos).detach().cpu().numpy(),
        )
        for model_name, scores in scores_per_model.items()
    ]
    cls = OrderedDict if isinstance(scores_per_model, OrderedDict) else dict
    scores_per_model_with_arrays = cls(scores_per_model_items)

    return utils_numpy.compare_models_pairwise_ttest(scores_per_model_with_arrays, alternative, higher_is_better)


compare_models_pairwise_ttest.__doc__ = compare_models_pairwise_ttest.__doc__.format(  # type: ignore[union-attr]
    docstring=utils_numpy.compare_models_pairwise_ttest.__doc__,
)


def compare_models_pairwise_wilcoxon(
    scores_per_model: dict[str, Tensor]
    | OrderedDict[str, Tensor]
    | dict[str, AUPIMOResult]
    | OrderedDict[str, AUPIMOResult],
    alternative: str,
    higher_is_better: bool,
) -> tuple[tuple[str, ...], dict[tuple[str, str], float]]:
    """Torch-oriented interface for `compare_models_pairwise_wilcoxon`. See its dscription for more details (below).

    Numpy version docstring
    =======================

    {docstring}
    """
    _validate_scores_per_model(scores_per_model)
    scores_per_model_items = [
        (
            model_name,
            (scores if isinstance(scores, Tensor) else scores.aupimos).detach().cpu().numpy(),
        )
        for model_name, scores in scores_per_model.items()
    ]
    cls = OrderedDict if isinstance(scores_per_model, OrderedDict) else dict
    scores_per_model_with_arrays = cls(scores_per_model_items)

    return utils_numpy.compare_models_pairwise_wilcoxon(scores_per_model_with_arrays, alternative, higher_is_better)


compare_models_pairwise_wilcoxon.__doc__ = compare_models_pairwise_wilcoxon.__doc__.format(  # type: ignore[union-attr]
    docstring=utils_numpy.compare_models_pairwise_wilcoxon.__doc__,
)


# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# NEXT
# GET RID OF THIS ERROR FOR NOT DEFINING AUPIMORESULT
# MAKE THE FORMATING OF STATS TEST LIKE BEFORE
