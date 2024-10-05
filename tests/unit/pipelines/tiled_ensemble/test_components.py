"""Test working of tiled ensemble pipeline components"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import copy
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from anomalib.data import get_datamodule
from anomalib.metrics import F1AdaptiveThreshold, ManualThreshold
from anomalib.pipelines.tiled_ensemble.components import MetricsCalculationJobGenerator, StatisticsJobGenerator, \
    SmoothingJobGenerator
from anomalib.pipelines.tiled_ensemble.components.smoothing import SmoothingJob
from anomalib.pipelines.tiled_ensemble.components.utils import NormalizationStage


class TestMerging:
    def test_tile_merging(self, get_ensemble_config, get_merging_mechanism):
        config = get_ensemble_config
        merger = get_merging_mechanism

        # prepared original data
        datamodule = get_datamodule(config)
        datamodule.prepare_data()
        datamodule.setup()
        original_data = next(iter(datamodule.test_dataloader()))

        batch = merger.ensemble_predictions.get_batch_tiles(0)

        merged_image = merger.merge_tiles(batch, "image")
        assert merged_image.equal(original_data["image"])

        merged_mask = merger.merge_tiles(batch, "mask")
        assert merged_mask.equal(original_data["mask"])

    def test_label_and_score_merging(self, get_merging_mechanism):
        merger = get_merging_mechanism
        scores = torch.rand(4, 10)
        labels = scores > 0.5

        mock_data = {(0, 0): {}, (0, 1): {}, (1, 0): {}, (1, 1): {}}

        for i, data in enumerate(mock_data.values()):
            data["pred_scores"] = scores[i]
            data["pred_labels"] = labels[i]

        merged = merger.merge_labels_and_scores(mock_data)

        assert merged["pred_scores"].equal(scores.mean(dim=0))

        assert merged["pred_labels"].equal(labels.any(dim=0))

    def test_box_merging(self, get_merging_mechanism):
        merger = get_merging_mechanism

        mock_data = {
            (0, 0): {
                "pred_boxes": [torch.ones(2, 4), torch.zeros(0, 4)],
                "box_scores": [torch.ones(2), torch.tensor([])],
                "box_labels": [torch.ones(2).type(torch.bool), torch.tensor([])],
            },
            (0, 1): {
                "pred_boxes": [torch.ones(1, 4), torch.ones(1, 4)],
                "box_scores": [torch.ones(1), torch.ones(1)],
                "box_labels": [torch.ones(1).type(torch.bool), torch.ones(1).type(torch.bool)],
            },
        }

        merged = merger.merge_boxes(mock_data)

        assert merged["pred_boxes"][0].shape == (3, 4)
        assert merged["box_scores"][0].shape == (3,)
        assert merged["box_labels"][0].shape == (3,)

        assert merged["pred_boxes"][1].shape == (1, 4)
        assert merged["box_scores"][1].shape == (1,)
        assert merged["box_labels"][1].shape == (1,)

    def test_box_merging_from_anomap(self, get_merging_mechanism):
        merger = get_merging_mechanism

        mock_anomaly_maps = torch.rand(2, 1, 50, 50)
        mock_anomaly_masks = mock_anomaly_maps > 0.5

        merged = merger.generate_boxes(mock_anomaly_maps, mock_anomaly_masks)

        assert "pred_boxes" in merged
        assert "box_scores" in merged
        assert "box_labels" in merged


class TestStatsCalculation:
    @pytest.mark.parametrize(
        "threshold_str, threshold_cls",
        (["F1AdaptiveThreshold", F1AdaptiveThreshold], ["ManualThreshold", ManualThreshold]),
    )
    def test_threshold_method(self, threshold_str, threshold_cls, get_ensemble_config):
        config = copy.deepcopy(get_ensemble_config)
        config["thresholding"]["method"] = threshold_str

        stats_job_generator = StatisticsJobGenerator(Path("mock"), threshold_str)
        stats_job = next(stats_job_generator.generate_jobs(None, None))

        assert isinstance(stats_job.image_threshold, threshold_cls)

    def test_stats(self, project_path):
        mock_preds = [
            {
                "pred_scores": torch.rand(4),
                "label": torch.ones(4),
                "box_scores": [torch.rand(1) for _ in range(4)],
                "anomaly_maps": torch.rand(4, 1, 50, 50),
                "mask": torch.ones(4, 1, 50, 50),
            },
        ]

        stats_job_generator = StatisticsJobGenerator(project_path, "F1AdaptiveThreshold")
        stats_job = next(stats_job_generator.generate_jobs(None, mock_preds))

        results = stats_job.run(None)

        assert "minmax" in results
        assert "image_threshold" in results
        assert "pixel_threshold" in results

        # save as it's removed from results
        save_path = results["save_path"]
        stats_job.save(results)
        assert Path(save_path).exists()


class TestMetrics:
    """Test ensemble metrics."""

    @pytest.fixture(scope="class")
    @staticmethod
    def get_ensemble_metrics_job(get_ensemble_config, get_batch_predictions):
        config = get_ensemble_config
        with TemporaryDirectory() as tmp_dir:
            metrics = MetricsCalculationJobGenerator(
                config["accelerator"],
                root_dir=Path(tmp_dir),
                task=config["data"]["init_args"]["task"],
                metrics=config["TrainModels"]["metrics"],
                normalization_stage=NormalizationStage(config["normalization_stage"]),
            )

        mock_predictions = get_batch_predictions

        return next(metrics.generate_jobs(prev_stage_result=copy.deepcopy(mock_predictions))), tmp_dir

    def test_metrics(self, get_ensemble_metrics_job):
        metrics_job, tmp_dir = get_ensemble_metrics_job

        result = metrics_job.run()

        assert "pixel_AUROC" in result
        assert "image_AUROC" in result

        metrics_job.save(result)
        assert (Path(tmp_dir) / "metric_results.csv").exists()


class TestJoinSmoothing:
    """Test JoinSmoothing job respnsible for smoothing area at tile seams."""

    @pytest.fixture(scope="class")
    def get_join_smoothing_job(self, get_ensemble_config, get_batch_predictions):
        config = get_ensemble_config
        job_gen = SmoothingJobGenerator(accelerator=config["accelerator"],
                                        tiling_args=config["tiling"],
                                        data_args=config["data"])
        # copy since smoothing changes data
        mock_predictions = copy.deepcopy(get_batch_predictions)
        job = next(job_gen.generate_jobs(config["SeamSmoothing"], mock_predictions))
        return job

    def test_mask(self, get_join_smoothing_job):
        smooth = get_join_smoothing_job

        join_index = smooth.tiler.tile_size_h, smooth.tiler.tile_size_w

        # join should be covered by True
        assert smooth.seam_mask[join_index]

        # non-join region should be false
        assert not smooth.seam_mask[0, 0]
        assert not smooth.seam_mask[-1, -1]

    def test_mask_overlapping(self, get_ensemble_config, get_batch_predictions):
        config = copy.deepcopy(get_ensemble_config)
        # tile size = 50, stride = 25 -> overlapping
        config["tiling"]["stride"] = 25
        job_gen = SmoothingJobGenerator(accelerator=config["accelerator"],
                                        tiling_args=config["tiling"],
                                        data_args=config["data"])
        # copy since smoothing changes data
        mock_predictions = copy.deepcopy(get_batch_predictions)
        smooth = next(job_gen.generate_jobs(config["SeamSmoothing"], mock_predictions))

        join_index = smooth.tiler.stride_h, smooth.tiler.stride_w

        # overlap join should be covered by True
        assert smooth.seam_mask[join_index]
        assert smooth.seam_mask[-join_index[0], -join_index[1]]

        # non-join region should be false
        assert not smooth.seam_mask[0, 0]
        assert not smooth.seam_mask[-1, -1]

    def test_smoothing(self, get_join_smoothing_job, get_batch_predictions):
        original_data = get_batch_predictions
        # fixture makes a copy of data
        smooth = get_join_smoothing_job

        # take first batch
        smoothed = smooth.run()[0]
        join_index = smooth.tiler.tile_size_h, smooth.tiler.tile_size_w

        # join sections should be processed
        assert not smoothed["anomaly_maps"][:, :, join_index].equal(original_data[0]["anomaly_maps"][:, :, join_index])

        # non-join section shouldn't be changed
        assert smoothed["anomaly_maps"][:, :, 0, 0].equal(original_data[0]["anomaly_maps"][:, :, 0, 0])

