"""Torch-oriented interfaces for `utils.py`."""

# Original Code
# https://github.com/jpcbertoldo/aupimo
#
# Modified
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import itertools
import logging
from collections import OrderedDict
from copy import deepcopy
from typing import TYPE_CHECKING

import matplotlib as mpl
import pandas as pd
import scipy
import scipy.stats
import torch
from pandas import DataFrame
from torch import Tensor

from . import _validate
from .enums import StatsAlternativeHypothesis, StatsOutliersPolicy, StatsRepeatedPolicy

if TYPE_CHECKING:
    from .pimo import AUPIMOResult


logger = logging.getLogger(__name__)


def per_image_scores_stats(
    per_image_scores: Tensor,
    images_classes: Tensor | None = None,
    only_class: int | None = None,
    outliers_policy: str | StatsOutliersPolicy = StatsOutliersPolicy.NONE.value,
    repeated_policy: str | StatsRepeatedPolicy = StatsRepeatedPolicy.AVOID.value,
    repeated_replacement_atol: float = 1e-2,
) -> list[dict[str, str | int | float]]:
    """Compute statistics of per-image scores (based on a boxplot's statistics).

    ***Torch-oriented interface for `.utils_numpy.per_image_scores_stats`***

    For a single per-image metric collection (1 model, 1 dataset), compute statistics (based on a boxplot)
    and find the closest image to each statistic.

    This function uses `matplotlib.cbook.boxplot_stats`, which is the same function used by `matplotlib.pyplot.boxplot`.

    ** OUTLIERS **
    Outliers are defined as in a boxplot, i.e. values that are more than 1.5 times the interquartile range (IQR) away
    from the Q1 and Q3 quartiles (respectively low and high outliers). The IQR is the difference between Q3 and Q1.

    Outliers are handled according to `outliers_policy`:
        - None | "none": do not include outliers.
        - "high": only include high outliers.
        - "low": only include low outliers.
        - "both": include both high and low outliers.

    ** IMAGE INDEX **
    Each statistic is associated with the image whose score is the closest to the statistic's value.

    ** REPEATED VALUES **
    It is possible that two stats have the same value (e.g. the median and the 25th percentile can be the same).
    Such cases are handled according to `repeated_policy`:
        - None | "none": do not address the issue, so several stats can have the same value and image index.
        - "avoid": avoid repeated values by iterativealy looking for other images with similar score, whose score
                    must be within `repeated_replacement_atol` (absolute tolerance) of the repeated value.

    Args:
        per_image_scores (Tensor): 1D Tensor of per-image scores.
        images_classes (Tensor | None):
            Used to filter statistics to only one class. If None, all images are considered.
            If given, 1D Tensor of binary image classes (0 for 'normal', 1 for 'anomalous'). Defaults to None.
        only_class (int | None):
            Only used if `images_classes` is not None.
            If not None, only compute statistics for images of the given class.
            `None` means both image classes are used.
            Defaults to None.
        outliers_policy (str | None): How to handle outliers stats (use them?). See `OutliersPolicy`. Defaults to None.
        repeated_policy (str | None): How to handle repeated values in boxplot stats (two stats with same value).
                                        See `RepeatedPolicy`. Defaults to None.
        repeated_replacement_atol (float): Absolute tolerance used to replace repeated values. Only used if
                                            `repeated_policy` is not None (or 'none'). Defaults to 1e-2 (1%).

    Returns:
        list[dict[str, str | int | float]]: List of boxplot statistics.

        Each dictionary has the following keys:
            - 'stat_name': Name of the statistic. Possible values:
                - 'mean': Mean of the scores.
                - 'med': Median of the scores.
                - 'q1': 25th percentile of the scores.
                - 'q3': 75th percentile of the scores.
                - 'whishi': Upper whisker value.
                - 'whislo': Lower whisker value.
                - 'outlo_i': low outlier value; `i` is a unique index for each low outlier.
                - 'outhi_j': high outlier value; `j` is a unique index for each high outlier.
            - 'stat_value': Value of the statistic (same units as `values`).
            - 'image_idx': Index of the image in `per_image_scores` whose score is the closest to the statistic's value.
            - 'score': The score of the image at index `image_idx` (not necessarily the same as `stat_value`).

        The list is sorted by increasing `stat_value`.
    """
    # other validations happen inside `utils_numpy.per_image_scores_stats`

    outliers_policy = StatsOutliersPolicy(outliers_policy)
    repeated_policy = StatsRepeatedPolicy(repeated_policy)
    _validate.is_per_image_scores(per_image_scores)

    # restrain the images to the class `only_class` if given, else use all images
    if images_classes is None:
        images_selection_mask = torch.ones_like(per_image_scores, dtype=bool)

    elif only_class is not None:
        _validate.is_images_classes(images_classes)
        _validate.is_same_shape(per_image_scores, images_classes)
        _validate.is_image_class(only_class)
        images_selection_mask = images_classes == only_class

    else:
        images_selection_mask = torch.ones_like(per_image_scores, dtype=bool)

    # indexes in `per_image_scores` are referred to as `candidate_idx`
    # while the indexes in the original array are referred to as `image_idx`
    #  - `candidate_idx` works for `per_image_scores` and `candidate2image_idx` (see below)
    #  - `image_idx` works for `images_classes` and `images_idxs_selected`
    per_image_scores = per_image_scores[images_selection_mask]
    # converts `candidate_idx` to `image_idx`
    candidate2image_idx = torch.nonzero(images_selection_mask, as_tuple=True)[0]

    # function used in `matplotlib.boxplot`
    boxplot_stats = mpl.cbook.boxplot_stats(per_image_scores)[0]  # [0] is for the only boxplot

    # remove unnecessary keys
    boxplot_stats = {name: value for name, value in boxplot_stats.items() if name not in {"iqr", "cilo", "cihi"}}

    # unroll `fliers` (outliers), remove unnecessary ones according to `outliers_policy`,
    # then add them to `boxplot_stats` with unique keys
    outliers = boxplot_stats.pop("fliers")
    outliers_lo = outliers[outliers < boxplot_stats["med"]]
    outliers_hi = outliers[outliers > boxplot_stats["med"]]

    if outliers_policy in {StatsOutliersPolicy.HIGH, StatsOutliersPolicy.BOTH}:
        boxplot_stats = {
            **boxplot_stats,
            **{f"outhi_{idx:06}": value for idx, value in enumerate(outliers_hi)},
        }

    if outliers_policy in {StatsOutliersPolicy.LOW, StatsOutliersPolicy.BOTH}:
        boxplot_stats = {
            **boxplot_stats,
            **{f"outlo_{idx:06}": value for idx, value in enumerate(outliers_lo)},
        }

    # state variables for the stateful function `append_record` below
    images_idxs_selected: set[int] = set()
    records: list[dict[str, str | int | float]] = []

    def append_record(stat_name: str, stat_value: float) -> None:
        candidates_sorted = torch.abs(per_image_scores - stat_value).argsort()
        candidate_idx = candidates_sorted[0]
        image_idx = candidate2image_idx[candidate_idx]

        # handle repeated values
        if image_idx not in images_idxs_selected or repeated_policy == StatsRepeatedPolicy.NONE:
            pass

        elif repeated_policy == StatsRepeatedPolicy.AVOID:
            for other_candidate_idx in candidates_sorted:
                other_candidate_image_idx = candidate2image_idx[other_candidate_idx]
                if other_candidate_image_idx in images_idxs_selected:
                    continue
                # if the code reaches here, it means that `other_candidate_image_idx` is not in `images_idxs_selected`
                # i.e. this image has not been selected yet, so it can be used
                other_candidate_score = per_image_scores[other_candidate_idx]
                # if the other candidate is not too far from the value, use it
                # note that the first choice has not changed, so if no other is selected in the loop
                # it will be the first choice
                if torch.isclose(other_candidate_score, stat_value, atol=repeated_replacement_atol):
                    candidate_idx = other_candidate_idx
                    image_idx = other_candidate_image_idx
                    break

        images_idxs_selected.add(image_idx)
        records.append(
            {
                "stat_name": stat_name,
                "stat_value": float(stat_value),
                "image_idx": int(image_idx),
                "score": float(per_image_scores[candidate_idx]),
            },
        )

    # loop over the stats from the lowest to the highest value
    for stat, val in sorted(boxplot_stats.items(), key=lambda x: x[1]):
        append_record(stat, val)
    return sorted(records, key=lambda r: r["score"])


def compare_models_pairwise_ttest_rel(
    scores_per_model: dict[str, Tensor]
    | OrderedDict[str, Tensor]
    | dict[str, "AUPIMOResult"]
    | OrderedDict[str, "AUPIMOResult"],
    alternative: str,
    higher_is_better: bool,
) -> tuple[tuple[str, ...], dict[tuple[str, str], float]]:
    """Compare all pairs of models using the paired t-test on two related samples (parametric).

    ***Torch-oriented interface for `.numpy_utils.compare_models_pairwise_ttest_rel`***

    This is a test for the null hypothesis that two repeated samples have identical average (expected) values.
    In fact, it tests whether the average of the differences between the two samples is significantly different from 0.

    Refs:
        - `scipy.stats.ttest_rel`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html
        - Wikipedia page: https://en.wikipedia.org/wiki/Student's_t-test#Dependent_t-test_for_paired_samples

    ===

    If an ordered dictionary is given, the models are sorted by the order of the dictionary.
    Otherwise, the models are sorted by average SCORE.

    Args:
        scores_per_model: Dictionary of `n` models and their per-image scores.
            key: model name
            value: tensor of shape (num_images,). All `nan` values must be at the same positions.
        higher_is_better: Whether higher values of score are better or worse. Defaults to True.
        alternative: Alternative hypothesis for the statistical tests. See `confidences` in "Returns" section.
                     Valid values are `StatsAlternativeHypothesis.ALTERNATIVES`.

    Returns:
        (models_ordered, test_results):
            - models_ordered: Models sorted by the user (`OrderedDict` input) or automatically (`dict` input).

                Automatic sorting is by average score from best to worst model.
                Depending on `higher_is_better`, this corresponds to:
                    - `higher_is_better=True` ==> descending score order
                    - `higher_is_better=False` ==> ascending score order
                along the indices from 0 to `n-1`.

            - confidences: Dictionary of confidence values for each pair of models.

                For all pairs of indices i and j from 0 to `n-1` such that i != j:
                    - key: (models_ordered[i], models_ordered[j])
                    - value: confidence on the alternative hypothesis.

                For models `models_ordered[i]` and `models_ordered[j]`, the alternative hypothesis is:
                    - if `less`: model[i] < model[j]
                    - if `greater`: model[i] > model[j]
                    - if `two-sided`: model[i] != model[j]
                in termos of average score.
    """
    _validate.is_scores_per_model(scores_per_model)
    scores_per_model_items = [
        (
            model_name,
            (scores if isinstance(scores, Tensor) else scores.aupimos),
        )
        for model_name, scores in scores_per_model.items()
    ]
    cls = OrderedDict if isinstance(scores_per_model, OrderedDict) else dict
    scores_per_model_with_arrays = cls(scores_per_model_items)

    _validate.is_scores_per_model(scores_per_model_with_arrays)
    StatsAlternativeHypothesis(alternative)

    # remove nan values; list of items keeps the order of the OrderedDict
    scores_per_model_nonan_items = [
        (model_name, scores[~torch.isnan(scores)]) for model_name, scores in scores_per_model_with_arrays.items()
    ]

    # sort models by average value if not an ordered dictionary
    # position 0 is assumed the best model
    if isinstance(scores_per_model_with_arrays, OrderedDict):
        scores_per_model_nonan = OrderedDict(scores_per_model_nonan_items)
    else:
        scores_per_model_nonan = OrderedDict(
            sorted(scores_per_model_nonan_items, key=lambda kv: kv[1].mean(), reverse=higher_is_better),
        )

    models_ordered = tuple(scores_per_model_nonan.keys())
    models_pairs = list(itertools.permutations(models_ordered, 2))
    confidences: dict[tuple[str, str], float] = {}
    for model_i, model_j in models_pairs:
        values_i = scores_per_model_nonan[model_i]
        values_j = scores_per_model_nonan[model_j]
        pvalue = scipy.stats.ttest_rel(
            values_i,
            values_j,
            alternative=alternative,
        ).pvalue
        confidences[model_i, model_j] = 1.0 - float(pvalue)

    return models_ordered, confidences


def compare_models_pairwise_wilcoxon(
    scores_per_model: dict[str, Tensor]
    | OrderedDict[str, Tensor]
    | dict[str, "AUPIMOResult"]
    | OrderedDict[str, "AUPIMOResult"],
    alternative: str,
    higher_is_better: bool,
    atol: float | None = 1e-3,
) -> tuple[tuple[str, ...], dict[tuple[str, str], float]]:
    """Compare all pairs of models using the Wilcoxon signed-rank test (non-parametric).

    ***Torch-oriented interface for `.numpy_utils.compare_models_pairwise_wilcoxon`***

    Each comparison of two models is a Wilcoxon signed-rank test (null hypothesis is that they are equal).

    It tests whether the distribution of the differences of scores is symmetric about zero in a non-parametric way.
    This is like the non-parametric version of the paired t-test.

    Refs:
        - `scipy.stats.wilcoxon`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html#scipy.stats.wilcoxon
        - Wikipedia page: https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test

    ===

    If an ordered dictionary is given, the models are sorted by the order of the dictionary.
    Otherwise, the models are sorted by average RANK.

    Args:
        scores_per_model: Dictionary of `n` models and their per-image scores.
            key: model name
            value: tensor of shape (num_images,). All `nan` values must be at the same positions.
        higher_is_better: Whether higher values of score are better or worse. Defaults to True.
        alternative: Alternative hypothesis for the statistical tests. See `confidences` in "Returns" section.
                     Valid values are `StatsAlternativeHypothesis.ALTERNATIVES`.
        atol: Absolute tolerance used to consider two scores as equal. Defaults to 1e-3 (0.1%).
              When doing a paired test, if the difference between two scores is below `atol`, the difference is
              truncated to 0. If `atol` is None, no truncation is done.

    Returns:
        (models_ordered, test_results):
            - models_ordered: Models sorted by the user (`OrderedDict` input) or automatically (`dict` input).

                Automatic sorting is from "best to worst" model, which corresponds to ascending average rank
                along the indices from 0 to `n-1`.

            - confidences: Dictionary of confidence values for each pair of models.

                For all pairs of indices i and j from 0 to `n-1` such that i != j:
                    - key: (models_ordered[i], models_ordered[j])
                    - value: confidence on the alternative hypothesis.

                For models `models_ordered[i]` and `models_ordered[j]`, the alternative hypothesis is:
                    - if `less`: model[i] < model[j]
                    - if `greater`: model[i] > model[j]
                    - if `two-sided`: model[i] != model[j]
                    in terms of average ranks (not scores!).
    """
    _validate.is_scores_per_model(scores_per_model)
    scores_per_model_items = [
        (
            model_name,
            (scores if isinstance(scores, Tensor) else scores.aupimos),
        )
        for model_name, scores in scores_per_model.items()
    ]
    cls = OrderedDict if isinstance(scores_per_model, OrderedDict) else dict
    scores_per_model_with_arrays = cls(scores_per_model_items)

    _validate.is_scores_per_model(scores_per_model_with_arrays)
    StatsAlternativeHypothesis(alternative)

    # remove nan values; list of items keeps the order of the OrderedDict
    scores_per_model_nonan_items = [
        (model_name, scores[~torch.isnan(scores)]) for model_name, scores in scores_per_model_with_arrays.items()
    ]

    # sort models by average value if not an ordered dictionary
    # position 0 is assumed the best model
    if isinstance(scores_per_model_with_arrays, OrderedDict):
        scores_per_model_nonan = OrderedDict(scores_per_model_nonan_items)
    else:
        # these average ranks will NOT consider `atol` because we want to rank the models anyway
        scores_nonan = torch.stack([v for _, v in scores_per_model_nonan_items], axis=0)
        avg_ranks = scipy.stats.rankdata(
            -scores_nonan if higher_is_better else scores_nonan,
            method="average",
            axis=0,
        ).mean(axis=1)
        # ascending order, lower score is better --> best to worst model
        argsort_avg_ranks = avg_ranks.argsort()
        scores_per_model_nonan = OrderedDict(scores_per_model_nonan_items[idx] for idx in argsort_avg_ranks)

    models_ordered = tuple(scores_per_model_nonan.keys())
    models_pairs = list(itertools.permutations(models_ordered, 2))
    confidences: dict[tuple[str, str], float] = {}
    for model_i, model_j in models_pairs:
        values_i = scores_per_model_nonan[model_i]
        values_j = scores_per_model_nonan[model_j]
        diff = values_i - values_j

        if atol is not None:
            # make the difference null if below the tolerance
            diff[torch.abs(diff) <= atol] = 0.0

        # extreme case
        if (diff == 0).all():  # noqa: SIM108
            pvalue = 1.0
        else:
            pvalue = scipy.stats.wilcoxon(diff, alternative=alternative).pvalue
        confidences[model_i, model_j] = 1.0 - float(pvalue)

    return models_ordered, confidences


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
    _validate.is_models_ordered(models_ordered)
    _validate.is_confidences(confidences)
    _validate.joint_validate_models_ordered_and_confidences(models_ordered, confidences)
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


def images_classes_from_masks(masks: torch.Tensor) -> torch.Tensor:
    """Deduce the image classes from the masks."""
    return (masks == 1).any(axis=(1, 2)).to(torch.int32)
