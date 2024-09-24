"""Enumerations for per-image metrics."""

# Based on the code: https://github.com/jpcbertoldo/aupimo

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class StatsOutliersPolicy(Enum):
    """How to handle outliers in per-image metrics boxplots. Use them? Only high? Only low? Both?

    Outliers are defined as in a boxplot, i.e. values that are more than 1.5 times the interquartile range (IQR) away
    from the Q1 and Q3 quartiles (respectively low and high outliers). The IQR is the difference between Q3 and Q1.

    None | "none": do not include outliers.
    "high": only include high outliers.
    "low": only include low outliers.
    "both": include both high and low outliers.
    """

    NONE: str = "none"
    HIGH: str = "high"
    LOW: str = "low"
    BOTH: str = "both"


class StatsRepeatedPolicy(Enum):
    """How to handle repeated values in per-image metrics boxplots (two stats with same value). Avoid them?

    None | "none": do not avoid repeated values, so several stats can have the same value and image index.
    "avoid": if a stat has the same value as another stat, the one with the closest then another image,
             with the nearest score, is selected.
    """

    NONE: str = "none"
    AVOID: str = "avoid"


class StatsAlternativeHypothesis(Enum):
    """Alternative hypothesis for the statistical tests used to compare per-image metrics."""

    TWO_SIDED: str = "two-sided"
    LESS: str = "less"
    GREATER: str = "greater"
