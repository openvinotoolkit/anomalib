"""Utility functions for per-image metrics.

author: jpcbertoldo
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from collections import OrderedDict
from typing import ClassVar

import matplotlib as mpl
import numpy as np
import scipy
import scipy.stats
from numpy import ndarray

from . import _validate

# =========================================== CONSTANTS ===========================================


class StatsOutliersPolicy:
    """How to handle outliers in per-image metrics boxplots. Use them? Only high? Only low? Both?

    Outliers are defined as in a boxplot, i.e. values that are more than 1.5 times the interquartile range (IQR) away
    from the Q1 and Q3 quartiles (respectively low and high outliers). The IQR is the difference between Q3 and Q1.

    None | "none": do not include outliers.
    "hi": only include high outliers.
    "lo": only include low outliers.
    "both": include both high and low outliers.
    """

    NONE: ClassVar[str] = "none"
    HI: ClassVar[str] = "hi"
    LO: ClassVar[str] = "lo"
    BOTH: ClassVar[str] = "both"

    POLICIES: ClassVar[tuple[str | None, ...]] = (None, NONE, HI, LO, BOTH)

    @staticmethod
    def validate(policy: str | None) -> None:
        """Validate the argument `policy`."""
        if policy not in StatsOutliersPolicy.POLICIES:
            msg = f"Invalid `policy`. Expected one of {StatsOutliersPolicy.POLICIES}, but got {policy}."
            raise ValueError(msg)


class StatsRepeatedPolicy:
    """How to handle repeated values in per-image metrics boxplots (two stats with same value). Avoid them?

    None | "none": do not avoid repeated values, so several stats can have the same value and image index.
    "avoid": if a stat has the same value as another stat, the one with the closest then another image,
             with the nearest score, is selected.
    """

    NONE: ClassVar[str] = "none"
    AVOID: ClassVar[str] = "avoid"

    POLICIES: ClassVar[tuple[str | None, ...]] = (None, NONE, AVOID)

    @staticmethod
    def validate(policy: str | None) -> None:
        """Validate the argument `policy`."""
        if policy not in StatsRepeatedPolicy.POLICIES:
            msg = f"Invalid `policy`. Expected one of {StatsRepeatedPolicy.POLICIES}, but got {policy}."
            raise ValueError(msg)


class StatsAlternativeHypothesis:
    """Alternative hypothesis for the statistical tests used to compare per-image metrics."""

    TWO_SIDED: ClassVar[str] = "two-sided"
    LESS: ClassVar[str] = "less"
    GREATER: ClassVar[str] = "greater"

    ALTERNATIVES: ClassVar[tuple[str, ...]] = (TWO_SIDED, LESS, GREATER)

    @staticmethod
    def validate(alternative: str) -> None:
        """Validate the argument `alternative`."""
        if alternative not in StatsAlternativeHypothesis.ALTERNATIVES:
            msg = (
                "Invalid `alternative`. "
                f"Expected one of {StatsAlternativeHypothesis.ALTERNATIVES}, but got {alternative}."
            )
            raise ValueError(msg)


# =========================================== ARGS VALIDATION ===========================================
def _validate_image_class(image_class: int) -> None:
    if not isinstance(image_class, int):
        msg = f"Expected image class to be an int (0 for 'normal', 1 for 'anomalous'), but got {type(image_class)}."
        raise TypeError(msg)

    if image_class not in (0, 1):
        msg = f"Expected image class to be either 0 for 'normal' or 1 for 'anomalous', but got {image_class}."
        raise ValueError(msg)


def _validate_per_image_scores(per_image_scores: ndarray) -> None:
    if not isinstance(per_image_scores, ndarray):
        msg = f"Expected per-image scores to be a numpy array, but got {type(per_image_scores)}."
        raise TypeError(msg)

    if per_image_scores.ndim != 1:
        msg = f"Expected per-image scores to be 1D, but got {per_image_scores.ndim}D."
        raise ValueError(msg)


def _validate_scores_per_model(scores_per_model: dict[str, ndarray] | OrderedDict[str, ndarray]) -> None:
    if not isinstance(scores_per_model, dict | OrderedDict):
        msg = f"Expected scores per model to be a dictionary or ordered dictionary, but got {type(scores_per_model)}."
        raise TypeError(msg)

    if len(scores_per_model) < 2:
        msg = f"Expected scores per model to have at least 2 models, but got {len(scores_per_model)}."
        raise ValueError(msg)

    first_key_value = None

    for model_name, scores in scores_per_model.items():
        if not isinstance(model_name, str):
            msg = f"Expected model name to be a string, but got {type(model_name)} for model {model_name}."
            raise TypeError(msg)

        if not isinstance(scores, ndarray):
            msg = f"Expected scores to be a numpy array, but got {type(scores)} for model {model_name}."
            raise TypeError(msg)

        if scores.ndim != 1:
            msg = f"Expected scores to be 1D, but got {scores.ndim}D for model {model_name}."
            raise ValueError(msg)

        num_valid_scores = scores[~np.isnan(scores)].shape[0]

        if num_valid_scores < 2:
            msg = f"Expected at least 2 scores, but got {num_valid_scores} for model {model_name}."
            raise ValueError(msg)

        if first_key_value is None:
            first_key_value = (model_name, scores)
            continue

        first_model_name, first_scores = first_key_value

        # same shape
        if scores.shape != first_scores.shape:
            msg = (
                "Expected scores to have the same shape, "
                f"but got ({model_name}) {scores.shape} != {first_scores.shape} ({first_model_name})."
            )
            raise ValueError(msg)

        # `nan` at the same indices
        if (np.isnan(scores) != np.isnan(first_scores)).any():
            msg = (
                "Expected `nan` values, if any, to be at the same indices, "
                f"but there are differences between models {model_name} and {first_model_name}."
            )
            raise ValueError(msg)


# =========================================== FUNCTIONS ===========================================


def per_image_scores_stats(
    per_image_scores: ndarray,
    images_classes: ndarray | None = None,
    only_class: int | None = None,
    outliers_policy: str | None = StatsOutliersPolicy.NONE,
    repeated_policy: str | None = StatsRepeatedPolicy.AVOID,
    repeated_replacement_atol: float = 1e-2,
) -> list[dict[str, str | int | float]]:
    """Compute statistics of per-image scores (based on a boxplot's statistics).

    For a single per-image metric collection (1 model, 1 dataset), compute statistics (based on a boxplot)
    and find the closest image to each statistic.

    This function uses `matplotlib.cbook.boxplot_stats`, which is the same function used by `matplotlib.pyplot.boxplot`.

    ** OUTLIERS **
    Outliers are defined as in a boxplot, i.e. values that are more than 1.5 times the interquartile range (IQR) away
    from the Q1 and Q3 quartiles (respectively low and high outliers). The IQR is the difference between Q3 and Q1.

    Outliers are handled according to `outliers_policy`:
        - None | "none": do not include outliers.
        - "hi": only include high outliers.
        - "lo": only include low outliers.
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
        per_image_scores (ndarray): 1D ndarray of per-image scores.
        images_classes (ndarray | None):
            Used to filter statistics to only one class. If None, all images are considered.
            If given, 1D ndarray of binary image classes (0 for 'normal', 1 for 'anomalous'). Defaults to None.
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
    StatsOutliersPolicy.validate(outliers_policy)
    StatsRepeatedPolicy.validate(repeated_policy)
    _validate_per_image_scores(per_image_scores)

    # restrain the images to the class `only_class` if given, else use all images
    if images_classes is None:
        images_selection_mask = np.ones_like(per_image_scores, dtype=bool)

    elif only_class is not None:
        _validate.images_classes(images_classes)
        _validate.same_shape(per_image_scores, images_classes)
        _validate_image_class(only_class)
        images_selection_mask = images_classes == only_class

    else:
        images_selection_mask = np.ones_like(per_image_scores, dtype=bool)

    # indexes in `per_image_scores` are referred to as `candidate_idx`
    # while the indexes in the original array are referred to as `image_idx`
    #  - `candidate_idx` works for `per_image_scores` and `candidate2image_idx` (see below)
    #  - `image_idx` works for `images_classes` and `images_idxs_selected`
    per_image_scores = per_image_scores[images_selection_mask]
    # converts `candidate_idx` to `image_idx`
    candidate2image_idx = np.nonzero(images_selection_mask)[0]

    # function used in `matplotlib.boxplot`
    boxplot_stats = mpl.cbook.boxplot_stats(per_image_scores)[0]  # [0] is for the only boxplot

    # remove unnecessary keys
    boxplot_stats = {name: value for name, value in boxplot_stats.items() if name not in ("iqr", "cilo", "cihi")}

    # unroll `fliers` (outliers), remove unnecessary ones according to `outliers_policy`,
    # then add them to `boxplot_stats` with unique keys
    outliers = boxplot_stats.pop("fliers")
    outliers_lo = outliers[outliers < boxplot_stats["med"]]
    outliers_hi = outliers[outliers > boxplot_stats["med"]]

    if outliers_policy in (StatsOutliersPolicy.HI, StatsOutliersPolicy.BOTH):
        boxplot_stats = {
            **boxplot_stats,
            **{f"outhi_{idx:06}": value for idx, value in enumerate(outliers_hi)},
        }

    if outliers_policy in (StatsOutliersPolicy.LO, StatsOutliersPolicy.BOTH):
        boxplot_stats = {
            **boxplot_stats,
            **{f"outlo_{idx:06}": value for idx, value in enumerate(outliers_lo)},
        }

    # state variables for the stateful function `append_record` below
    images_idxs_selected: set[int] = set()
    records: list[dict[str, str | int | float]] = []

    def append_record(stat_name: str, stat_value: float) -> None:
        candidates_sorted = np.abs(per_image_scores - stat_value).argsort()
        candidate_idx = candidates_sorted[0]
        image_idx = candidate2image_idx[candidate_idx]

        # handle repeated values
        if image_idx not in images_idxs_selected or repeated_policy is None:
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
                if np.isclose(other_candidate_score, stat_value, atol=repeated_replacement_atol):
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
    scores_per_model: dict[str, ndarray] | OrderedDict[str, ndarray],
    alternative: str,
    higher_is_better: bool,
) -> tuple[tuple[str, ...], dict[tuple[str, str], float]]:
    """Compare all pairs of models using the paired t-test on two related samples (parametric).

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
    _validate_scores_per_model(scores_per_model)
    StatsAlternativeHypothesis.validate(alternative)

    # remove nan values; list of items keeps the order of the OrderedDict
    scores_per_model_nonan_items = [
        (model_name, scores[~np.isnan(scores)]) for model_name, scores in scores_per_model.items()
    ]

    # sort models by average value if not an ordered dictionary
    # position 0 is assumed the best model
    if isinstance(scores_per_model, OrderedDict):
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
        confidences[(model_i, model_j)] = 1.0 - float(pvalue)

    return models_ordered, confidences


def compare_models_pairwise_wilcoxon(
    scores_per_model: dict[str, ndarray] | OrderedDict[str, ndarray],
    alternative: str,
    higher_is_better: bool,
    atol: float | None = 1e-3,
) -> tuple[tuple[str, ...], dict[tuple[str, str], float]]:
    """Compare all pairs of models using the Wilcoxon signed-rank test (non-parametric).

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
    _validate_scores_per_model(scores_per_model)
    StatsAlternativeHypothesis.validate(alternative)

    # remove nan values; list of items keeps the order of the OrderedDict
    scores_per_model_nonan_items = [
        (model_name, scores[~np.isnan(scores)]) for model_name, scores in scores_per_model.items()
    ]

    # sort models by average value if not an ordered dictionary
    # position 0 is assumed the best model
    if isinstance(scores_per_model, OrderedDict):
        scores_per_model_nonan = OrderedDict(scores_per_model_nonan_items)
    else:
        # these average ranks will NOT consider `atol` because we want to rank the models anyway
        scores_nonan = np.stack([v for _, v in scores_per_model_nonan_items], axis=0)
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
            diff[np.abs(diff) <= atol] = 0.0

        # extreme case
        if (diff == 0).all():  # noqa: SIM108
            pvalue = 1.0
        else:
            pvalue = scipy.stats.wilcoxon(diff, alternative=alternative).pvalue
        confidences[(model_i, model_j)] = 1.0 - float(pvalue)

    return models_ordered, confidences
