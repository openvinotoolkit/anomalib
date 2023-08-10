"""Test all pipeline functions"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest

from tools.tiled_ensemble.post_processing import (
    SmoothJoins,
    PostProcessStats,
    Threshold,
    MinMaxNormalize,
    EnsembleVisualization,
    EnsembleMetrics,
)
from tools.tiled_ensemble.post_processing.pipelines import (
    get_stats_pipeline,
    get_stats,
    get_postprocessing_pipeline,
    post_process,
)


class TestStatsPipeline:
    @pytest.mark.parametrize("smooth, present", ([True, True], [True, True]))
    def test_stats_smooth(self, smooth, present, get_ens_config, get_tiler):
        config = copy.deepcopy(get_ens_config)
        tiler = get_tiler

        config.ensemble.post_processing.smooth_joins.apply = smooth
        config.ensemble.metrics.threshold.method = "adaptive"
        pipe = get_stats_pipeline(config, tiler)

        assert isinstance(pipe.steps[0], SmoothJoins) == present

    @pytest.mark.parametrize("threshold, t_present", (["adaptive", True], ["manual", False], ["none", False]))
    @pytest.mark.parametrize("normalization, n_present", (["tile", False], ["final", True]))
    def test_stats_pipeline(self, threshold, t_present, normalization, n_present, get_ens_config, get_tiler):
        config = copy.deepcopy(get_ens_config)
        tiler = get_tiler

        config.ensemble.metrics.threshold.method = threshold
        config.ensemble.post_processing.normalization = normalization

        pipe = get_stats_pipeline(config, tiler)

        # if either threshold requires it or norm requires it, it should be present
        assert isinstance(pipe.steps[-1], PostProcessStats) == (t_present or n_present)

    def test_get_stats(self, get_ens_config, get_tiler, get_ensemble_predictions):
        config = get_ens_config
        tiler = get_tiler
        val_pred = get_ensemble_predictions

        stats = get_stats(config, tiler, copy.deepcopy(val_pred))

        assert "min" in stats
        assert "max" in stats
        assert "image_threshold" in stats
        assert "pixel_threshold" in stats


mock_stats = {
    "min": 0,
    "max": 1,
    "image_threshold": 0.42,
    "pixel_threshold": 0.13,
}


class TestPostProcess:
    @pytest.mark.parametrize("smooth, present", ([True, True], [True, True]))
    def test_stats_smooth(self, smooth, present, get_ens_config, get_tiler):
        config = copy.deepcopy(get_ens_config)
        tiler = get_tiler
        stats = copy.deepcopy(mock_stats)

        config.ensemble.post_processing.smooth_joins.apply = smooth
        pipe = get_postprocessing_pipeline(config, tiler, stats)

        assert isinstance(pipe.steps[0], SmoothJoins) == present

    @pytest.mark.parametrize(
        "normalization, thresholds, threshold_index", (["final", [0.5, 0.5], 2], ["none", [42, 13], 1])
    )
    def test_threshold_manual(self, normalization, thresholds, threshold_index, get_ens_config, get_tiler):
        config = copy.deepcopy(get_ens_config)
        tiler = get_tiler
        stats = copy.deepcopy(mock_stats)

        config.ensemble.metrics.threshold.method = "manual"
        config.ensemble.post_processing.normalization = normalization

        config.ensemble.metrics.threshold.manual_image = 42
        config.ensemble.metrics.threshold.manual_pixel = 13

        pipe = get_postprocessing_pipeline(config, tiler, stats)

        # if minmax present -> 0.5, else unchanged
        assert pipe.steps[threshold_index].image_threshold == thresholds[0]
        assert pipe.steps[threshold_index].pixel_threshold == thresholds[1]

    def test_step_order(self, get_ens_config, get_tiler):
        config = get_ens_config
        tiler = get_tiler
        stats = copy.deepcopy(mock_stats)

        pipe = get_postprocessing_pipeline(config, tiler, stats)

        steps = pipe.steps

        assert isinstance(steps[0], SmoothJoins)
        assert isinstance(steps[1], MinMaxNormalize)
        assert isinstance(steps[2], Threshold)
        assert isinstance(steps[3], EnsembleVisualization)
        assert isinstance(steps[4], EnsembleMetrics)

    def test_postprocess_pipeline(self, get_ens_config, get_tiler, get_ensemble_predictions):
        config = get_ens_config
        tiler = get_tiler
        val_pred = get_ensemble_predictions
        ens_pred = get_ensemble_predictions

        original = val_pred.get_batch_tiles(0)[(0, 0)]["anomaly_maps"].clone()

        pipe_out = post_process(config, tiler, validation_predictions=val_pred, ensemble_predictions=ens_pred)

        after_pipe = val_pred.get_batch_tiles(0)[(0, 0)]["anomaly_maps"]

        # stats pipeline shouldn't change data
        assert original.equal(after_pipe)
        assert list(pipe_out.keys()) == ["image_F1Score", "image_AUROC", "pixel_F1Score", "pixel_AUROC"]
