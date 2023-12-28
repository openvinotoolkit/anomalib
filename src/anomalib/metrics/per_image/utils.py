"""Torch-oriented interfaces for `utils.py`."""
from __future__ import annotations

import logging
import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import TYPE_CHECKING

import pandas as pd
import torch
from pandas import DataFrame
from torch import Tensor

from . import _validate, utils_numpy
from .utils_numpy import StatsOutliersPolicy, StatsRepeatedPolicy

if TYPE_CHECKING:
    from .pimo import AUPIMOResult


logger = logging.getLogger(__name__)

# =========================================== ARGS VALIDATION ===========================================


def _validate_models_ordered(models_ordered: tuple[str, ...]) -> None:
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


def _validate_confidences(confidences: dict[tuple[str, str], float]) -> None:
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


def _joint_validate_models_ordered_and_confidences(
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


# =========================================== FUNCTIONS ===========================================


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


def _validate_scores_per_model_tensor(scores_per_model: dict[str, Tensor] | OrderedDict[str, Tensor]) -> None:
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


def _validate_scores_per_model_aupimoresult(
    scores_per_model: dict[str, AUPIMOResult] | OrderedDict[str, AUPIMOResult],
    missing_paths_ok: bool,
) -> None:
    first_key_value = None

    for model_name, aupimoresult in scores_per_model.items():
        if first_key_value is None:
            first_key_value = (model_name, aupimoresult)
            continue

        first_model_name, first_aupimoresult = first_key_value

        # check that the metadata is the same, so they can be compared indeed
        if aupimoresult.shared_fpr_metric != first_aupimoresult.shared_fpr_metric:
            msg = (
                "Expected AUPIMOResult objects in scores per model to have the same shared FPR metric, "
                f"but got ({model_name}) {aupimoresult.shared_fpr_metric} != "
                f"{first_aupimoresult.shared_fpr_metric} ({first_model_name})."
            )
            raise ValueError(msg)

        if aupimoresult.fpr_bounds != first_aupimoresult.fpr_bounds:
            msg = (
                "Expected AUPIMOResult objects in scores per model to have the same FPR bounds, "
                f"but got ({model_name}) {aupimoresult.fpr_bounds} != "
                f"{first_aupimoresult.fpr_bounds} ({first_model_name})."
            )
            raise ValueError(msg)

    available_paths = [tuple(scores.paths) for scores in scores_per_model.values() if scores.paths is not None]

    if len(set(available_paths)) > 1:
        msg = (
            "Expected AUPIMOResult objects in scores per model to have the same paths, "
            "but got different paths for different models."
        )
        raise ValueError(msg)

    if len(available_paths) != len(scores_per_model):
        msg = "Some models have paths, while others are missing them."
        if missing_paths_ok:
            warnings.warn(msg, UserWarning, stacklevel=3)
            logger.warning(msg)


def _validate_scores_per_model(
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
        _validate_scores_per_model_tensor(scores_per_model)
        return

    _validate_scores_per_model_tensor(
        {model_name: scores.aupimos for model_name, scores in scores_per_model.items()},
    )

    _validate_scores_per_model_aupimoresult(scores_per_model, missing_paths_ok=True)


def compare_models_pairwise_ttest_rel(
    scores_per_model: dict[str, Tensor]
    | OrderedDict[str, Tensor]
    | dict[str, AUPIMOResult]
    | OrderedDict[str, AUPIMOResult],
    alternative: str,
    higher_is_better: bool,
) -> tuple[tuple[str, ...], dict[tuple[str, str], float]]:
    """Torch-oriented interface for `compare_models_pairwise_ttest_rel`. See its dscription for more details (below).

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

    return utils_numpy.compare_models_pairwise_ttest_rel(scores_per_model_with_arrays, alternative, higher_is_better)


compare_models_pairwise_ttest_rel.__doc__ = compare_models_pairwise_ttest_rel.__doc__.format(  # type: ignore[union-attr]
    docstring=utils_numpy.compare_models_pairwise_ttest_rel.__doc__,
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


def format_pairwise_tests_results(
    models_ordered: tuple[str, ...],
    confidences: dict[tuple[str, str], float],
    model1_as_column: bool = True,
    left_to_right: bool = False,
    top_to_bottom: bool = False,
) -> DataFrame:
    """Format the results of pairwise tests into a square dataframe.

    The confidence values refer to the confidence level (in [0, 1]) on the alternative hypothesis,
    which is formulated as "`model1` <alternative> `model2`", where `<alternative>` can be '<', '>', or '!='.

    HOW TO READ THE DATAFRAME
    =========================
    There are 6 possible ways to read the dataframe, depending on the values of `model1_as_column` and `alternative`
    (from the pairwise test function that generated `confidences`).

    *column* and *row* below refer to a generic column and row value (model names) in the dataframe.

    if (
        model1_as_column == True and alternative == 'less'
        or model1_as_column == False and alternative == 'greater'
    )
        read: "column < row"
        equivalently: "row > column"

    elif (
        model1_as_column == True and alternative == 'greater'
        or model1_as_column == False and alternative == 'less'
    )
        read: "column > row"
        equivalently: "row < column"

    else:  # alternative == 'two-sided'
        read: "column != row"
        equivalently: "row != column"

    Args:
        models_ordered: The models ordered in a meaningful way, generally from best to worst when automatically ordered.
        confidences: The confidence on the alternative hypothesis, as returned by the pairwise test function.
        model1_as_column: Whether to put `model1` as column or row in the dataframe.
        left_to_right: Whether to order the columns from best to worst model as left to right.
        top_to_bottom: Whether to order the rows from best to worst model as top to bottom.
            Default column/row ordering is from worst to best model (left to right, top to bottom),
            so the upper left corner is the worst model compared to itself, and the bottom right corner is the best
            model compared to itself.

    """
    _validate_models_ordered(models_ordered)
    _validate_confidences(confidences)
    _joint_validate_models_ordered_and_confidences(models_ordered, confidences)
    confidences = deepcopy(confidences)
    confidences.update({(model, model): torch.nan for model in models_ordered})
    # `df` stands for `dataframe`
    confdf = pd.DataFrame(confidences, index=["confidence"]).T
    confdf.index.names = ["model1", "model2"]
    confdf = confdf.reset_index()
    confdf["model1"] = pd.Categorical(confdf["model1"], categories=models_ordered, ordered=True)
    confdf["model2"] = pd.Categorical(confdf["model2"], categories=models_ordered, ordered=True)
    # df at this point: 3 columns: model1, model2, confidence
    index_model, column_model = ("model2", "model1") if model1_as_column else ("model1", "model2")
    confdf = confdf.pivot_table(index=index_model, columns=column_model, values="confidence", dropna=False, sort=False)
    # now it is a square dataframe with models as index and columns, and confidence as values
    confdf = confdf.sort_index(axis=0, ascending=top_to_bottom)
    return confdf.sort_index(axis=1, ascending=left_to_right)
