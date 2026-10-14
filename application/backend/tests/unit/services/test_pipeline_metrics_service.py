# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import statistics
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from pydantic_models.metrics import PipelineMetrics
from pydantic_models.model import Model
from pydantic_models.pipeline import Pipeline, PipelineStatus
from pydantic_models.project import Project
from pydantic_models.sink import FolderSinkConfig, OutputFormat, SinkType
from pydantic_models.source import SourceType, VideoFileSourceConfig
from services.metrics_service import MetricsService
from services.pipeline_metrics_service import PipelineMetricsService
from services.pipeline_service import PipelineService


@pytest.fixture
def fxt_project() -> Project:
    """Fixture for a test project."""
    return Project(
        id=uuid4(),
        name="Test Project",
    )


@pytest.fixture
def fxt_model(fxt_project) -> Model:
    """Fixture for a test model."""
    id = uuid4()
    return Model(
        id=id,
        project_id=fxt_project.id,
        name=f"Test Model ({str(id).split('-')[0]})",
        architecture="padim",
        train_job_id=uuid4(),
        dataset_snapshot_id=uuid4(),
    )


@pytest.fixture
def fxt_pipeline(fxt_project, fxt_model) -> Pipeline:
    """Fixture for a test pipeline."""
    source = VideoFileSourceConfig(
        id=uuid4(),
        project_id=fxt_project.id,
        source_type=SourceType.VIDEO_FILE,
        name="Test Source",
        video_path="/path/to/video.mp4",
    )
    sink = FolderSinkConfig(
        id=uuid4(),
        project_id=fxt_project.id,
        sink_type=SinkType.FOLDER,
        name="Test Sink",
        folder_path="/path/to/output",
        output_formats=[OutputFormat.IMAGE_ORIGINAL],
        rate_limit=0.2,
    )
    return Pipeline(
        project_id=fxt_project.id,
        source=source,
        sink=sink,
        model=fxt_model,
        source_id=source.id,
        sink_id=sink.id,
        model_id=fxt_model.id,
        status=PipelineStatus.RUNNING,
    )


@pytest.fixture
def fxt_idle_pipeline(fxt_project) -> Pipeline:
    """Fixture for an idle pipeline."""
    return Pipeline(project_id=fxt_project.id, status=PipelineStatus.IDLE)


@pytest.fixture
def fxt_metrics_service():
    """Fixture for a mock MetricsService."""
    return MagicMock(spec=MetricsService)


@pytest.fixture
def fxt_pipeline_service():
    """Fixture for a mock PipelineService."""
    return MagicMock(spec=PipelineService)


@pytest.fixture
def fxt_pipeline_metrics_service(fxt_pipeline_service, fxt_metrics_service):
    """Fixture for PipelineMetricsService with mocked dependencies."""
    return PipelineMetricsService(
        pipeline_service=fxt_pipeline_service,
        metrics_service=fxt_metrics_service,
    )


class TestPipelineMetricsService:
    @pytest.mark.parametrize(
        "data,percentile,expected",
        [
            ([], 95, 0.0),
            ([1.0], 50, 1.0),
            ([42.0], 95, 42.0),  # Single value edge case
            ([1.0, 2.0, 3.0, 4.0, 5.0], 0, 1.0),  # 0th percentile edge case
            ([1.0, 2.0, 3.0, 4.0, 5.0], 50, 3.0),
            ([1.0, 2.0, 3.0, 4.0, 5.0], 90, 4.6),
            ([1.0, 2.0, 3.0, 4.0, 5.0], 95, 4.8),
            ([1.0, 2.0, 3.0, 4.0, 5.0], 100, 5.0),  # 100th percentile edge case
        ],
    )
    def test_calculate_percentile(self, data, percentile, expected):
        """Test percentile calculation with various inputs including edge cases."""
        result = PipelineMetricsService._calculate_percentile(data, percentile)
        assert abs(result - expected) < 0.01  # Allow small floating point differences

    @pytest.mark.parametrize(
        "latency_samples,time_window,expected_metrics,should_raise",
        [
            ([10.5, 12.3, 8.7, 15.2, 9.1], 60, "with_data", False),
            ([5.1, 7.3, 6.8, 8.2, 4.9], 120, "with_data", False),
            ([], 30, "no_data", False),
            (None, 60, None, True),  # Not running pipeline
        ],
    )
    def test_get_pipeline_metrics(
        self,
        fxt_pipeline_metrics_service,
        fxt_pipeline,
        fxt_idle_pipeline,
        fxt_pipeline_service,
        fxt_metrics_service,
        latency_samples,
        time_window,
        expected_metrics,
        should_raise,
    ):
        """Test getting pipeline metrics with different scenarios."""
        if should_raise:
            # Use idle pipeline for not running test
            pipeline = fxt_idle_pipeline
            fxt_metrics_service.get_latency_measurements.return_value = latency_samples
        else:
            pipeline = fxt_pipeline
            fxt_metrics_service.get_latency_measurements.return_value = latency_samples
            # Mock throughput measurements (returns tuple: total_requests, throughput_data)
            # throughput_data is a list of (timestamp, count) tuples
            if latency_samples:
                throughput_data = [(i, 1.0) for i in range(len(latency_samples))]
                fxt_metrics_service.get_throughput_measurements.return_value = (len(latency_samples), throughput_data)
            else:
                fxt_metrics_service.get_throughput_measurements.return_value = (0, [])

        fxt_pipeline_service.get_pipeline_by_id = AsyncMock(return_value=pipeline)

        if should_raise:
            with pytest.raises(ValueError, match="Cannot get metrics for a pipeline that is not running"):
                asyncio.run(fxt_pipeline_metrics_service.get_pipeline_metrics(pipeline.project_id))
        else:
            result = asyncio.run(fxt_pipeline_metrics_service.get_pipeline_metrics(pipeline.project_id, time_window))

            assert isinstance(result, PipelineMetrics)
            assert result.time_window.time_window == time_window

            if expected_metrics == "with_data":
                assert result.inference.latency.avg_ms == statistics.mean(latency_samples)
                assert result.inference.latency.min_ms == min(latency_samples)
                assert result.inference.latency.max_ms == max(latency_samples)
                assert result.inference.latency.latest_ms == latency_samples[-1]
                assert result.inference.throughput.avg_requests_per_second is not None
                assert result.inference.throughput.total_requests is not None
                fxt_metrics_service.get_latency_measurements.assert_called_once_with(
                    model_id=pipeline.model_id,
                    time_window=time_window,
                )
            elif expected_metrics == "no_data":
                assert result.inference.latency.avg_ms is None
                assert result.inference.latency.min_ms is None
                assert result.inference.latency.max_ms is None
                assert result.inference.latency.p95_ms is None
                assert result.inference.latency.latest_ms is None
                assert result.inference.throughput.avg_requests_per_second is None
                assert result.inference.throughput.total_requests is None
