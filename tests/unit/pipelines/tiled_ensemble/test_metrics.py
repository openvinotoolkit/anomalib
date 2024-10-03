"""Test ensemble metrics"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from anomalib.metrics import AnomalibMetricCollection
from anomalib.pipelines.tiled_ensemble.components import MetricsCalculationJobGenerator
from anomalib.pipelines.tiled_ensemble.components.utils import NormalizationStage

mock_predictions = [
    {
        "pred_scores": torch.ones(50),
        "label": torch.ones(50),
        "anomaly_maps": torch.ones(10, 50),
        "mask": torch.ones(10, 50),
    },
]


@pytest.fixture(scope="module")
def get_ensemble_metrics_job(get_ensemble_config):
    config = get_ensemble_config
    with TemporaryDirectory() as tmp_dir:
        metrics = MetricsCalculationJobGenerator(
            config["accelerator"],
            root_dir=Path(tmp_dir),
            task=config["data"]["init_args"]["task"],
            metrics=config["TrainModels"]["metrics"],
            normalization_stage=NormalizationStage(config["normalization_stage"]),
        )

    return next(metrics.generate_jobs(prev_stage_result=copy.deepcopy(mock_predictions))), tmp_dir


def test_metrics_setup(get_ensemble_metrics_job):
    metrics_job, _ = get_ensemble_metrics_job

    assert isinstance(metrics_job.image_metrics, AnomalibMetricCollection)
    assert isinstance(metrics_job.pixel_metrics, AnomalibMetricCollection)


def test_data_unchanged(get_ensemble_metrics_job):
    metrics_job, _ = get_ensemble_metrics_job

    metrics_job.run()

    for name, values in metrics_job.predictions[0].items():
        assert values.equal(mock_predictions[0][name]), f"{name} changed"


def test_metric_save(get_ensemble_metrics_job):
    metrics_job, tmp_dir = get_ensemble_metrics_job

    result = metrics_job.run()
    metrics_job.save(result)

    assert (Path(tmp_dir) / "metric_results.csv").exists()
