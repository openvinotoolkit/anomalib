"""Test post-processing elements"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch

from tools.tiled_ensemble.post_processing import (
    EnsemblePostProcessPipeline,
    MinMaxNormalize,
    PostProcessStats,
    SmoothJoins,
    Threshold,
)

mock_data = {
    "image": torch.rand((5, 3, 100, 100)),
    "mask": torch.zeros((5, 100, 100)),
    "anomaly_maps": torch.rand((5, 1, 100, 100)),
    "label": torch.zeros(5),
    "pred_scores": torch.ones(5),
    "pred_labels": torch.ones(5),
    "pred_masks": torch.zeros((5, 100, 100)),
    "pred_boxes": [torch.rand(1, 4) for _ in range(5)],
    "box_labels": [torch.tensor([0.5]) for _ in range(5)],
    "box_scores": [torch.tensor([0.5]) for _ in range(5)],
}


@pytest.fixture()
def get_smooth_joins(get_tiler, get_ensemble_config):
    config = get_ensemble_config
    tiler = get_tiler

    return SmoothJoins(
        width_factor=config.ensemble.post_processing.smooth_joins.width,
        filter_sigma=config.ensemble.post_processing.smooth_joins.sigma,
        tiler=tiler,
    )


class TestSmoothJoins:
    def test_mask(self, get_smooth_joins):
        smooth = get_smooth_joins

        join_index = smooth.tiler.tile_size_h, smooth.tiler.tile_size_w

        # join should be covered by True
        assert smooth.join_mask[join_index]

        # non-join region should be false
        assert not smooth.join_mask[0, 0]

    def test_smoothing(self, get_smooth_joins):
        smooth = get_smooth_joins

        smoothed = smooth.process(copy.deepcopy(mock_data))
        join_index = smooth.tiler.tile_size_h, smooth.tiler.tile_size_w

        # join sections should be processed
        assert not smoothed["anomaly_maps"][:, :, join_index].equal(mock_data["anomaly_maps"][:, :, join_index])

        # non-join section shouldn't be changed
        assert smoothed["anomaly_maps"][:, :, 0, 0].equal(mock_data["anomaly_maps"][:, :, 0, 0])


class TestStats:
    @pytest.mark.parametrize(
        "key, value",
        [
            ("anomaly_maps", torch.rand(5, 1, 50, 50)),
            ("box_scores", [torch.rand(1) for _ in range(5)]),
            ("pred_scores", torch.rand(5)),
        ],
    )
    def test_minmax(self, key, value):
        stats = PostProcessStats()

        # remove given keys to test all possible sources of minmax
        data = {"pred_scores": torch.rand(5), "label": torch.ones(5), key: value}

        stats.process(data)
        result = stats.compute()

        if isinstance(value, list):
            value = torch.cat(value)

        assert result["min"] == torch.min(value)
        assert result["max"] == torch.max(value)

    @pytest.mark.parametrize(
        ["labels", "preds", "target_threshold"],
        [
            (torch.Tensor([0, 0, 0, 1, 1]), torch.Tensor([2.3, 1.6, 2.6, 7.9, 3.3]), 3.3),  # standard case
            (torch.Tensor([1, 0, 0, 0]), torch.Tensor([4, 3, 2, 1]), 4),  # 100% recall for all thresholds
        ],
    )
    def test_threshold(self, labels, preds, target_threshold):
        data = {
            "label": labels,
            "mask": labels,
            "pred_scores": preds,
            "anomaly_maps": preds,
        }

        stats = PostProcessStats()

        stats.process(data)
        result = stats.compute()

        assert round(result["image_threshold"], 5) == target_threshold
        assert round(result["pixel_threshold"], 5) == target_threshold


class TestMinMaxNormalize:
    @staticmethod
    def get_stats(data):
        stats_block = PostProcessStats()
        stats_block.process(data)
        return stats_block.compute()

    def test_normalization(self):
        stats = self.get_stats(mock_data)

        normalization = MinMaxNormalize(stats)

        normalized = normalization.process(copy.deepcopy(mock_data))

        norm_stats = self.get_stats(normalized)

        assert norm_stats["image_threshold"] == 0.5
        assert norm_stats["pixel_threshold"] == 0.5


class TestThreshold:
    def test_score_threshold(self):
        threshold_block = Threshold(0.5, 0.5)

        data = {"pred_scores": torch.tensor([0.7, 0.8, 0.1, 0.33, 0.5])}

        thresholded = threshold_block.process(data)

        assert thresholded["pred_labels"].equal(torch.tensor([True, True, False, False, True]))

    def test_mask_threshold(self):
        threshold_block = Threshold(0.5, 0.5)

        data = {
            "pred_scores": torch.tensor([0.7, 0.8, 0.1, 0.33, 0.5]),
            "anomaly_maps": torch.tensor([[0.7, 0.8, 0.1], [0.33, 0.5, 0.1]]),
        }

        thresholded = threshold_block.process(data)

        assert thresholded["pred_masks"].equal(torch.tensor([[True, True, False], [False, True, False]]))
        assert "pred_boxes" in thresholded
        assert "box_scores" in thresholded
        assert "box_labels" in thresholded

    def test_box_threshold(self):
        threshold_block = Threshold(0.5, 0.5)

        data = {
            "pred_scores": torch.tensor([0.7, 0.8, 0.1, 0.33, 0.5]),
            "box_scores": [torch.tensor([0.6]), torch.tensor([0.1, 0.69])],
        }

        thresholded = threshold_block.process(data)

        assert "box_labels" in thresholded

        assert thresholded["box_labels"][0].equal(torch.tensor([True]))
        assert thresholded["box_labels"][1].equal(torch.tensor([False, True]))


class TestPostProcessPipeline:
    def test_execute(self, get_ensemble_config, get_ensemble_predictions, get_joiner, get_smooth_joins):
        joiner = get_joiner
        data = get_ensemble_predictions
        smooth_joins = get_smooth_joins

        pipeline = EnsemblePostProcessPipeline(joiner)

        pipeline.add_steps([smooth_joins, Threshold(0.5, 0.5)])

        pipeline.execute(data)

    def test_results(self, get_joiner, get_ensemble_predictions, get_ensemble_metrics):
        pipeline = EnsemblePostProcessPipeline(get_joiner)

        pipeline.add_steps([Threshold(0.5, 0.5), get_ensemble_metrics])

        pipe_out = pipeline.execute(get_ensemble_predictions)

        assert "metrics" in pipe_out
