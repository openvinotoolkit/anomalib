"""Test working of tiled ensemble pipeline components."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from anomalib.data import get_datamodule
from anomalib.metrics import F1AdaptiveThreshold, ManualThreshold
from anomalib.pipelines.tiled_ensemble.components import (
    MergeJobGenerator,
    MetricsCalculationJobGenerator,
    NormalizationJobGenerator,
    SmoothingJobGenerator,
    StatisticsJobGenerator,
    ThresholdingJobGenerator,
)
from anomalib.pipelines.tiled_ensemble.components.metrics_calculation import MetricsCalculationJob
from anomalib.pipelines.tiled_ensemble.components.smoothing import SmoothingJob
from anomalib.pipelines.tiled_ensemble.components.utils import NormalizationStage
from anomalib.pipelines.tiled_ensemble.components.utils.prediction_data import EnsemblePredictions
from anomalib.pipelines.tiled_ensemble.components.utils.prediction_merging import PredictionMergingMechanism


class TestMerging:
    """Test merging mechanism and merging job."""

    @staticmethod
    def test_tile_merging(get_ensemble_config: dict, get_merging_mechanism: PredictionMergingMechanism) -> None:
        """Test tiled data merging."""
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

    @staticmethod
    def test_label_and_score_merging(get_merging_mechanism: PredictionMergingMechanism) -> None:
        """Test label and score merging."""
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

    @staticmethod
    def test_merge_job(
        get_tile_predictions: EnsemblePredictions,
        get_ensemble_config: dict,
        get_merging_mechanism: PredictionMergingMechanism,
    ) -> None:
        """Test merging job execution."""
        config = get_ensemble_config
        predictions = copy.deepcopy(get_tile_predictions)
        merging_mechanism = get_merging_mechanism

        merging_job_generator = MergeJobGenerator(tiling_args=config["tiling"], data_args=config["data"])
        merging_job = next(merging_job_generator.generate_jobs(prev_stage_result=predictions))

        merged_direct = merging_mechanism.merge_tile_predictions(0)
        merged_with_job = merging_job.run()[0]

        # check that merging by job is same as with the mechanism directly
        for key, value in merged_direct.items():
            if isinstance(value, torch.Tensor):
                assert merged_with_job[key].equal(value)
            elif isinstance(value, list) and isinstance(value[0], torch.Tensor):
                # boxes
                assert all(j.equal(d) for j, d in zip(merged_with_job[key], value, strict=False))
            else:
                assert merged_with_job[key] == value


class TestStatsCalculation:
    """Test post-processing statistics calculations."""

    @staticmethod
    @pytest.mark.parametrize(
        ("threshold_str", "threshold_cls"),
        [("F1AdaptiveThreshold", F1AdaptiveThreshold), ("ManualThreshold", ManualThreshold)],
    )
    def test_threshold_method(threshold_str: str, threshold_cls: type, get_ensemble_config: dict) -> None:
        """Test that correct thresholding method is used."""
        config = copy.deepcopy(get_ensemble_config)
        config["thresholding"]["method"] = threshold_str

        stats_job_generator = StatisticsJobGenerator(Path("mock"), threshold_str)
        stats_job = next(stats_job_generator.generate_jobs(None, None))

        assert isinstance(stats_job.image_threshold, threshold_cls)

    @staticmethod
    def test_stats_run(project_path: Path) -> None:
        """Test execution of statistics calc. job."""
        mock_preds = [
            {
                "pred_scores": torch.rand(4),
                "label": torch.ones(4),
                "anomaly_maps": torch.rand(4, 1, 50, 50),
                "mask": torch.ones(4, 1, 50, 50),
            },
        ]

        stats_job_generator = StatisticsJobGenerator(project_path, "F1AdaptiveThreshold")
        stats_job = next(stats_job_generator.generate_jobs(None, mock_preds))

        results = stats_job.run()

        assert "minmax" in results
        assert "image_threshold" in results
        assert "pixel_threshold" in results

        # save as it's removed from results
        save_path = results["save_path"]
        stats_job.save(results)
        assert Path(save_path).exists()

    @staticmethod
    @pytest.mark.parametrize(
        ("key", "values"),
        [
            ("anomaly_maps", [torch.rand(5, 1, 50, 50), torch.rand(5, 1, 50, 50)]),
            ("pred_scores", [torch.rand(5), torch.rand(5)]),
        ],
    )
    def test_minmax(key: str, values: list) -> None:
        """Test minmax stats calculation."""
        # add given keys to test all possible sources of minmax
        data = [
            {"pred_scores": torch.rand(5), "label": torch.ones(5), key: values[0]},
            {"pred_scores": torch.rand(5), "label": torch.ones(5), key: values[1]},
        ]

        stats_job_generator = StatisticsJobGenerator(Path("mock"), "F1AdaptiveThreshold")
        stats_job = next(stats_job_generator.generate_jobs(None, data))
        results = stats_job.run()

        if isinstance(values[0], list):
            values[0] = torch.cat(values[0])
            values[1] = torch.cat(values[1])
        values = torch.stack(values)

        assert results["minmax"][key]["min"] == torch.min(values)
        assert results["minmax"][key]["max"] == torch.max(values)

    @staticmethod
    @pytest.mark.parametrize(
        ("labels", "preds", "target_threshold"),
        [
            (torch.Tensor([0, 0, 0, 1, 1]), torch.Tensor([2.3, 1.6, 2.6, 7.9, 3.3]), 3.3),  # standard case
            (torch.Tensor([1, 0, 0, 0]), torch.Tensor([4, 3, 2, 1]), 4),  # 100% recall for all thresholds
        ],
    )
    def test_threshold(labels: torch.Tensor, preds: torch.Tensor, target_threshold: float) -> None:
        """Test threshold calculation job."""
        data = [
            {
                "label": labels,
                "mask": labels,
                "pred_scores": preds,
                "anomaly_maps": preds,
            },
        ]

        stats_job_generator = StatisticsJobGenerator(Path("mock"), "F1AdaptiveThreshold")
        stats_job = next(stats_job_generator.generate_jobs(None, data))
        results = stats_job.run()

        assert round(results["image_threshold"], 5) == target_threshold
        assert round(results["pixel_threshold"], 5) == target_threshold


class TestMetrics:
    """Test ensemble metrics."""

    @pytest.fixture(scope="class")
    @staticmethod
    def get_ensemble_metrics_job(
        get_ensemble_config: dict,
        get_batch_predictions: list[dict],
    ) -> tuple[MetricsCalculationJob, str]:
        """Return Metrics calculation job and path to directory where metrics csv will be saved."""
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

    @staticmethod
    def test_metrics_result(get_ensemble_metrics_job: tuple[MetricsCalculationJob, str]) -> None:
        """Test metrics result."""
        metrics_job, _ = get_ensemble_metrics_job

        result = metrics_job.run()

        assert "pixel_AUROC" in result
        assert "image_AUROC" in result

    @staticmethod
    def test_metrics_saving(get_ensemble_metrics_job: tuple[MetricsCalculationJob, str]) -> None:
        """Test metrics saving to csv."""
        metrics_job, tmp_dir = get_ensemble_metrics_job

        result = metrics_job.run()
        metrics_job.save(result)
        assert (Path(tmp_dir) / "metric_results.csv").exists()


class TestJoinSmoothing:
    """Test JoinSmoothing job responsible for smoothing area at tile seams."""

    @pytest.fixture(scope="class")
    @staticmethod
    def get_join_smoothing_job(get_ensemble_config: dict, get_batch_predictions: list[dict]) -> SmoothingJob:
        """Make and return SmoothingJob instance."""
        config = get_ensemble_config
        job_gen = SmoothingJobGenerator(
            accelerator=config["accelerator"],
            tiling_args=config["tiling"],
            data_args=config["data"],
        )
        # copy since smoothing changes data
        mock_predictions = copy.deepcopy(get_batch_predictions)
        return next(job_gen.generate_jobs(config["SeamSmoothing"], mock_predictions))

    @staticmethod
    def test_mask(get_join_smoothing_job: SmoothingJob) -> None:
        """Test seam mask in case where tiles don't overlap."""
        smooth = get_join_smoothing_job

        join_index = smooth.tiler.tile_size_h, smooth.tiler.tile_size_w

        # seam should be covered by True
        assert smooth.seam_mask[join_index]

        # non-seam region should be false
        assert not smooth.seam_mask[0, 0]
        assert not smooth.seam_mask[-1, -1]

    @staticmethod
    def test_mask_overlapping(get_ensemble_config: dict, get_batch_predictions: list[dict]) -> None:
        """Test seam mask in case where tiles overlap."""
        config = copy.deepcopy(get_ensemble_config)
        # tile size = 50, stride = 25 -> overlapping
        config["tiling"]["stride"] = 25
        job_gen = SmoothingJobGenerator(
            accelerator=config["accelerator"],
            tiling_args=config["tiling"],
            data_args=config["data"],
        )
        mock_predictions = copy.deepcopy(get_batch_predictions)
        smooth = next(job_gen.generate_jobs(config["SeamSmoothing"], mock_predictions))

        join_index = smooth.tiler.stride_h, smooth.tiler.stride_w

        # overlap seam should be covered by True
        assert smooth.seam_mask[join_index]
        assert smooth.seam_mask[-join_index[0], -join_index[1]]

        # non-seam region should be false
        assert not smooth.seam_mask[0, 0]
        assert not smooth.seam_mask[-1, -1]

    @staticmethod
    def test_smoothing(get_join_smoothing_job: SmoothingJob, get_batch_predictions: list[dict]) -> None:
        """Test smoothing job run."""
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


def test_normalization(get_batch_predictions: list[dict], project_path: Path) -> None:
    """Test normalization step."""
    original_predictions = copy.deepcopy(get_batch_predictions)

    for batch in original_predictions:
        batch["anomaly_maps"] *= 100
        batch["pred_scores"] *= 100

    # # get and save stats using stats job on predictions
    stats_job_generator = StatisticsJobGenerator(project_path, "F1AdaptiveThreshold")
    stats_job = next(stats_job_generator.generate_jobs(prev_stage_result=original_predictions))
    stats = stats_job.run()
    stats_job.save(stats)

    # normalize predictions based on obtained stats
    norm_job_generator = NormalizationJobGenerator(root_dir=project_path)
    # copy as this changes preds
    norm_job = next(norm_job_generator.generate_jobs(prev_stage_result=original_predictions))
    normalized_predictions = norm_job.run()

    for batch in normalized_predictions:
        assert (batch["anomaly_maps"] >= 0).all()
        assert (batch["anomaly_maps"] <= 1).all()

        assert (batch["pred_scores"] >= 0).all()
        assert (batch["pred_scores"] <= 1).all()


class TestThresholding:
    """Test tiled ensemble thresholding stage."""

    @pytest.fixture(scope="class")
    @staticmethod
    def get_threshold_job(get_mock_stats_dir: Path) -> callable:
        """Return a function that takes prediction data and runs threshold job."""
        thresh_job_generator = ThresholdingJobGenerator(
            root_dir=get_mock_stats_dir,
            normalization_stage=NormalizationStage.IMAGE,
        )

        def thresh_helper(preds: dict) -> list | None:
            thresh_job = next(thresh_job_generator.generate_jobs(prev_stage_result=preds))
            return thresh_job.run()

        return thresh_helper

    @staticmethod
    def test_score_threshold(get_threshold_job: callable) -> None:
        """Test anomaly score thresholding."""
        thresholding = get_threshold_job

        data = [{"pred_scores": torch.tensor([0.7, 0.8, 0.1, 0.33, 0.5])}]

        thresholded = thresholding(data)[0]

        assert thresholded["pred_labels"].equal(torch.tensor([True, True, False, False, True]))

    @staticmethod
    def test_anomap_threshold(get_threshold_job: callable) -> None:
        """Test anomaly map thresholding."""
        thresholding = get_threshold_job

        data = [
            {
                "pred_scores": torch.tensor([0.7, 0.8, 0.1, 0.33, 0.5]),
                "anomaly_maps": torch.tensor([[0.7, 0.8, 0.1], [0.33, 0.5, 0.1]]),
            },
        ]

        thresholded = thresholding(data)[0]

        assert thresholded["pred_masks"].equal(torch.tensor([[True, True, False], [False, True, False]]))
