"""Test `utils.py`."""

# Original Code
# https://github.com/jpcbertoldo/aupimo
#
# Modified
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from anomalib.metrics.pimo import (
    StatsOutliersPolicy,
    StatsRepeatedPolicy,
    per_image_scores_stats,
)


def test_per_image_scores_stats() -> None:
    """Test `per_image_scores_boxplot_stats`."""
    gen = torch.Generator().manual_seed(42)
    num_scores = 201
    scores = torch.randn(num_scores, generator=gen)

    stats = per_image_scores_stats(scores)
    assert len(stats) == 6
    for statdic in stats:
        assert "stat_name" in statdic
        stat_name = statdic["stat_name"]
        assert stat_name in {"mean", "med", "q1", "q3", "whishi", "whislo"} or stat_name.startswith(
            ("outlo_", "outhi_"),
        )
        assert "stat_value" in statdic
        assert "image_idx" in statdic
        image_idx = statdic["image_idx"]
        assert 0 <= image_idx <= num_scores - 1

    classes = (torch.arange(num_scores) % 3 == 0).to(torch.long)
    stats = per_image_scores_stats(scores, classes, only_class=None)
    assert len(stats) == 6
    stats = per_image_scores_stats(scores, classes, only_class=0)
    assert len(stats) == 6
    stats = per_image_scores_stats(scores, classes, only_class=1)
    assert len(stats) == 6

    stats = per_image_scores_stats(scores, outliers_policy=StatsOutliersPolicy.BOTH)
    assert len(stats) == 6
    stats = per_image_scores_stats(scores, outliers_policy=StatsOutliersPolicy.LOW)
    assert len(stats) == 6
    stats = per_image_scores_stats(scores, outliers_policy=StatsOutliersPolicy.HIGH)
    assert len(stats) == 6
    stats = per_image_scores_stats(scores, outliers_policy=StatsOutliersPolicy.NONE)
    assert len(stats) == 6

    # force repeated values
    scores = torch.round(scores * 10) / 10
    stats = per_image_scores_stats(scores, repeated_policy=StatsRepeatedPolicy.AVOID)
    assert len(stats) == 6
    stats = per_image_scores_stats(
        scores,
        classes,
        repeated_policy=StatsRepeatedPolicy.AVOID,
        repeated_replacement_atol=1e-1,
    )
    assert len(stats) == 6
    stats = per_image_scores_stats(scores, repeated_policy=StatsRepeatedPolicy.NONE)
    assert len(stats) == 6


def test_per_image_scores_stats_specific_values() -> None:
    """Test `per_image_scores_boxplot_stats` with specific values."""
    scores = torch.concatenate(
        [
            # whislo = min value is 0.0
            torch.tensor([0.0]),
            torch.zeros(98),
            # q1 value is 0.0
            torch.tensor([0.0]),
            torch.linspace(0.01, 0.29, 98),
            # med value is 0.3
            torch.tensor([0.3]),
            torch.linspace(0.31, 0.69, 99),
            # q3 value is 0.7
            torch.tensor([0.7]),
            torch.linspace(0.71, 0.99, 99),
            # whishi = max value is 1.0
            torch.tensor([1.0]),
        ],
    )

    stats = per_image_scores_stats(scores)
    assert len(stats) == 6

    statdict_whislo = stats[0]
    statdict_q1 = stats[1]
    statdict_med = stats[2]
    statdict_mean = stats[3]
    statdict_q3 = stats[4]
    statdict_whishi = stats[5]

    assert statdict_whislo["stat_name"] == "whislo"
    assert np.isclose(statdict_whislo["stat_value"], 0.0)

    assert statdict_q1["stat_name"] == "q1"
    assert np.isclose(statdict_q1["stat_value"], 0.0, atol=1e-2)

    assert statdict_med["stat_name"] == "med"
    assert np.isclose(statdict_med["stat_value"], 0.3, atol=1e-2)

    assert statdict_mean["stat_name"] == "mean"
    assert np.isclose(statdict_mean["stat_value"], 0.3762, atol=1e-2)

    assert statdict_q3["stat_name"] == "q3"
    assert np.isclose(statdict_q3["stat_value"], 0.7, atol=1e-2)

    assert statdict_whishi["stat_name"] == "whishi"
    assert statdict_whishi["stat_value"] == 1.0
