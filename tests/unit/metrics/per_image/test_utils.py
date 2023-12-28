"""Test `utils.py`."""

from collections import OrderedDict

import numpy as np
import pytest
import torch
from torch import Tensor

from anomalib.metrics.per_image import AUPIMOResult, PIMOSharedFPRMetric


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Generate test cases."""
    num_images = 100
    # avg is 0.8
    aucs1 = 0.8 * torch.ones(num_images)
    # avg ~ 0.7
    aucs2 = torch.linspace(0.6, 0.8, num_images)
    # avg ~ 0.6
    aucs3 = torch.sin(torch.linspace(0, torch.pi, num_images)).clip(0, 1)

    mock_aupimoresult_stuff = {
        "shared_fpr_metric": PIMOSharedFPRMetric.MEAN_PERIMAGE_FPR,
        "fpr_lower_bound": 1e-5,
        "fpr_upper_bound": 1e-4,
        "num_threshs": 1_000,
        "thresh_lower_bound": 1.0,
        "thresh_upper_bound": 2.0,
    }
    scores_per_model_dicts = [
        ({"a": aucs1, "b": aucs2},),
        ({"a": aucs1, "b": aucs2, "c": aucs3},),
        (OrderedDict([("c", aucs1), ("b", aucs2), ("a", aucs3)]),),
        (
            {
                "a": AUPIMOResult(**{**mock_aupimoresult_stuff, "aupimos": aucs1}),
                "b": AUPIMOResult(**{**mock_aupimoresult_stuff, "aupimos": aucs2}),
                "c": AUPIMOResult(**{**mock_aupimoresult_stuff, "aupimos": aucs3}),
            },
        ),
    ]

    if (
        metafunc.function is test_compare_models_pairwise_ttest
        or metafunc.function is test_compare_models_pairwise_wilcoxon
    ):
        metafunc.parametrize(("scores_per_model",), scores_per_model_dicts)
        metafunc.parametrize(
            ("alternative", "higher_is_better"),
            [
                ("two-sided", True),
                ("two-sided", False),
                ("less", False),
                ("greater", True),
                # not considering the case (less, true) and (greater, false) because it will break
                # some assumptions in the assertions but they are possible
            ],
        )

    if metafunc.function is test_format_pairwise_tests_results:
        metafunc.parametrize(("scores_per_model",), scores_per_model_dicts[:3])


def assert_statsdict_stuff(statdic: dict, max_image_idx: int) -> None:
    """Assert stuff about a `statdic`."""
    assert "stat_name" in statdic
    stat_name = statdic["stat_name"]
    assert stat_name in ("mean", "med", "q1", "q3", "whishi", "whislo") or stat_name.startswith(
        ("outlo_", "outhi_"),
    )
    assert "stat_value" in statdic
    assert "image_idx" in statdic
    image_idx = statdic["image_idx"]
    assert 0 <= image_idx <= max_image_idx


def test_per_image_scores_stats() -> None:
    """Test `per_image_scores_boxplot_stats`."""
    from anomalib.metrics.per_image import (
        StatsOutliersPolicy,
        StatsRepeatedPolicy,
        per_image_scores_stats,
    )

    gen = torch.Generator().manual_seed(42)
    num_scores = 201
    scores = torch.randn(num_scores, generator=gen)

    stats = per_image_scores_stats(scores)
    assert len(stats) == 6
    for statdic in stats:
        assert_statsdict_stuff(statdic, num_scores - 1)

    classes = (torch.arange(num_scores) % 3 == 0).to(torch.long)
    stats = per_image_scores_stats(scores, classes, only_class=None)
    assert len(stats) == 6
    stats = per_image_scores_stats(scores, classes, only_class=0)
    assert len(stats) == 6
    stats = per_image_scores_stats(scores, classes, only_class=1)
    assert len(stats) == 6

    stats = per_image_scores_stats(scores, outliers_policy=StatsOutliersPolicy.BOTH)
    assert len(stats) == 6
    stats = per_image_scores_stats(scores, outliers_policy=StatsOutliersPolicy.LO)
    assert len(stats) == 6
    stats = per_image_scores_stats(scores, outliers_policy=StatsOutliersPolicy.HI)
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
    from anomalib.metrics.per_image import per_image_scores_stats

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


def test_compare_models_pairwise_ttest(scores_per_model: dict, alternative: str, higher_is_better: bool) -> None:
    """Test `compare_models_pairwise_ttest`."""
    from anomalib.metrics.per_image import AUPIMOResult, compare_models_pairwise_ttest

    models_ordered, confidences = compare_models_pairwise_ttest(
        scores_per_model,
        alternative=alternative,
        higher_is_better=higher_is_better,
    )
    assert len(confidences) == (len(models_ordered) * (len(models_ordered) - 1))

    diff = set(scores_per_model.keys()).symmetric_difference(set(models_ordered))
    assert len(diff) == 0

    if isinstance(scores_per_model, OrderedDict):
        assert models_ordered == tuple(scores_per_model.keys())

    elif len(scores_per_model) == 2:
        assert models_ordered == (("a", "b") if higher_is_better else ("b", "a"))

    elif len(scores_per_model) == 3:
        assert models_ordered == (("a", "b", "c") if higher_is_better else ("c", "b", "a"))

    if isinstance(next(iter(scores_per_model.values())), AUPIMOResult):
        return

    def copy_and_add_nan(scores: Tensor) -> Tensor:
        scores = scores.clone()
        scores[5:] = torch.nan
        return scores

    # removing samples should reduce the confidences
    scores_per_model["a"] = copy_and_add_nan(scores_per_model["a"])
    scores_per_model["b"] = copy_and_add_nan(scores_per_model["b"])
    if "c" in scores_per_model:
        scores_per_model["c"] = copy_and_add_nan(scores_per_model["c"])

    compare_models_pairwise_ttest(
        scores_per_model,
        alternative=alternative,
        higher_is_better=higher_is_better,
    )


def test_compare_models_pairwise_wilcoxon(scores_per_model: dict, alternative: str, higher_is_better: bool) -> None:
    """Test `compare_models_pairwise_wilcoxon`."""
    from anomalib.metrics.per_image import AUPIMOResult, compare_models_pairwise_wilcoxon

    models_ordered, confidences = compare_models_pairwise_wilcoxon(
        scores_per_model,
        alternative=alternative,
        higher_is_better=higher_is_better,
    )
    assert len(confidences) == (len(models_ordered) * (len(models_ordered) - 1))

    diff = set(scores_per_model.keys()).symmetric_difference(set(models_ordered))
    assert len(diff) == 0

    if isinstance(scores_per_model, OrderedDict):
        assert models_ordered == tuple(scores_per_model.keys())

    elif len(scores_per_model) == 2:
        assert models_ordered == (("a", "b") if higher_is_better else ("b", "a"))

    elif len(scores_per_model) == 3:
        # this one is not trivial without looking at the data, so no assertions
        pass

    if isinstance(next(iter(scores_per_model.values())), AUPIMOResult):
        return

    def copy_and_add_nan(scores: Tensor) -> Tensor:
        scores = scores.clone()
        scores[5:] = torch.nan
        return scores

    # removing samples should reduce the confidences
    scores_per_model["a"] = copy_and_add_nan(scores_per_model["a"])
    scores_per_model["b"] = copy_and_add_nan(scores_per_model["b"])
    if "c" in scores_per_model:
        scores_per_model["c"] = copy_and_add_nan(scores_per_model["c"])

    compare_models_pairwise_wilcoxon(
        scores_per_model,
        alternative=alternative,
        higher_is_better=higher_is_better,
    )


def test_format_pairwise_tests_results(scores_per_model: dict) -> None:
    """Test `format_pairwise_tests_results`."""
    from anomalib.metrics.per_image import (
        compare_models_pairwise_ttest,
        compare_models_pairwise_wilcoxon,
        format_pairwise_tests_results,
    )

    models_ordered, confidences = compare_models_pairwise_wilcoxon(
        scores_per_model,
        alternative="greater",
        higher_is_better=True,
    )
    confidence_df = format_pairwise_tests_results(
        models_ordered,
        confidences,
        model1_as_column=True,
        left_to_right=True,
        top_to_bottom=True,
    )
    assert tuple(confidence_df.columns.tolist()) == models_ordered
    assert tuple(confidence_df.index.tolist()) == models_ordered

    models_ordered, confidences = compare_models_pairwise_ttest(
        scores_per_model,
        alternative="greater",
        higher_is_better=True,
    )
    confidence_df = format_pairwise_tests_results(
        models_ordered,
        confidences,
        model1_as_column=True,
        left_to_right=True,
        top_to_bottom=True,
    )
    assert tuple(confidence_df.columns.tolist()) == models_ordered
    assert tuple(confidence_df.index.tolist()) == models_ordered
