"""Torch-oriented interfaces for `utils.py`."""

# Original Code
# https://github.com/jpcbertoldo/aupimo
#
# Modified
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import matplotlib as mpl
import torch
from torch import Tensor

from . import _validate
from .enums import StatsOutliersPolicy, StatsRepeatedPolicy

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


def images_classes_from_masks(masks: torch.Tensor) -> torch.Tensor:
    """Deduce the image classes from the masks."""
    return (masks == 1).any(axis=(1, 2)).to(torch.int32)
