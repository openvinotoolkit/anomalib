# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import status
from pydantic import ValidationError

from api.dependencies import get_pipeline_metrics_service, get_pipeline_service
from main import app
from pydantic_models.metrics import InferenceMetrics, LatencyMetrics, PipelineMetrics, ThroughputMetrics, TimeWindow
from pydantic_models.model import Model
from pydantic_models.pipeline import Pipeline, PipelineStatus
from pydantic_models.sink import FolderSinkConfig
from pydantic_models.source import VideoFileSourceConfig
from services import ActivePipelineConflictError, PipelineService
from services.pipeline_metrics_service import PipelineMetricsService


@pytest.fixture
def fxt_pipeline() -> Pipeline:
    return Pipeline(
        project_id=uuid4(),
        status=PipelineStatus.IDLE,
    )


@pytest.fixture
def fxt_pipeline_service() -> MagicMock:
    pipeline_service = MagicMock(spec=PipelineService)
    # get_active_pipeline is an async static method, so we need to mock it as AsyncMock
    pipeline_service.get_active_pipeline = AsyncMock(return_value=None)
    app.dependency_overrides[get_pipeline_service] = lambda: pipeline_service
    return pipeline_service


@pytest.fixture
def fxt_pipeline_metrics_service() -> MagicMock:
    pipeline_metrics_service = MagicMock(spec=PipelineMetricsService)
    app.dependency_overrides[get_pipeline_metrics_service] = lambda: pipeline_metrics_service
    return pipeline_metrics_service


class TestPipelineEndpoints:
    @pytest.mark.parametrize(
        "http_method, service_method",
        [
            ("get", "get_pipeline_by_id"),
            ("patch", "update_pipeline"),
        ],
    )
    def test_pipeline_invalid_ids(self, http_method, service_method, fxt_pipeline_service, fxt_client):
        response = getattr(fxt_client, http_method)("/api/projects/invalid-id/pipeline")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        getattr(fxt_pipeline_service, service_method).assert_not_called()

    def test_get_pipeline_success(self, fxt_pipeline, fxt_pipeline_service, fxt_client):
        fxt_pipeline_service.get_pipeline_by_id.return_value = fxt_pipeline

        response = fxt_client.get(f"/api/projects/{fxt_pipeline.project_id}/pipeline")

        assert response.status_code == status.HTTP_200_OK
        fxt_pipeline_service.get_pipeline_by_id.assert_called_once_with(fxt_pipeline.project_id)

    def test_update_pipeline_success(self, fxt_pipeline, fxt_pipeline_service, fxt_client):
        project_id, sink_id = fxt_pipeline.project_id, str(uuid4())
        fxt_pipeline_service.update_pipeline.return_value = fxt_pipeline

        response = fxt_client.patch(f"/api/projects/{project_id}/pipeline", json={"sink_id": sink_id})

        assert response.status_code == status.HTTP_200_OK
        fxt_pipeline_service.update_pipeline.assert_called_once_with(project_id, {"sink_id": sink_id})

    def test_update_pipeline_status(self, fxt_pipeline, fxt_pipeline_service, fxt_client):
        response = fxt_client.patch(
            f"/api/projects/{fxt_pipeline.project_id}/pipeline",
            json={"status": PipelineStatus.IDLE},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_pipeline_service.update_pipeline.assert_not_called()

    @pytest.mark.parametrize(
        "operation, pipeline_status",
        [
            ("run", PipelineStatus.RUNNING),
            ("disable", PipelineStatus.IDLE),
        ],
    )
    def test_enable_pipeline(self, operation, pipeline_status, fxt_pipeline, fxt_pipeline_service, fxt_client):
        project_id = fxt_pipeline.project_id
        # Mock get_active_pipeline to return None (no active pipeline) for run operation
        if operation == "run":
            fxt_pipeline_service.get_active_pipeline = AsyncMock(return_value=None)
            # activate_pipeline internally calls update_pipeline, so we make activate_pipeline call update_pipeline

            async def activate_pipeline_side_effect(proj_id, set_running=False):
                await fxt_pipeline_service.update_pipeline(proj_id, {"status": pipeline_status})
                return fxt_pipeline

            fxt_pipeline_service.activate_pipeline = AsyncMock(side_effect=activate_pipeline_side_effect)
            fxt_pipeline_service.update_pipeline = AsyncMock(return_value=fxt_pipeline)
        response = fxt_client.post(f"/api/projects/{project_id}/pipeline:{operation}")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        if operation == "run":
            fxt_pipeline_service.activate_pipeline.assert_called_once_with(project_id, set_running=True)
            fxt_pipeline_service.update_pipeline.assert_called_once_with(project_id, {"status": pipeline_status})
        else:
            fxt_pipeline_service.update_pipeline.assert_called_once_with(project_id, {"status": pipeline_status})

    @pytest.mark.parametrize("operation", ["run", "disable"])
    def test_enable_pipeline_invalid_id(self, operation, fxt_pipeline, fxt_pipeline_service, fxt_client):
        response = fxt_client.post(f"/api/projects/invalid-id/pipeline:{operation}")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_pipeline_service.update_pipeline.assert_not_called()

    # Note: ResourceNotFoundError handling not implemented in current endpoints
    # def test_enable_non_existent_pipeline(
    #     self, pipeline_op, pipeline_status, fxt_pipeline, fxt_pipeline_service, fxt_client
    # ):
    #     project_id = fxt_pipeline.project_id
    #     fxt_pipeline_service.update_pipeline.side_effect = ResourceNotFoundError(
    #         resource_type=ResourceType.PIPELINE,
    #         resource_id=str(project_id),
    #     )
    #     response = fxt_client.post(f"/api/projects/{project_id}/pipeline:{pipeline_op}")
    #     assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    #     fxt_pipeline_service.update_pipeline.assert_called_once_with(project_id, {"status": pipeline_status})

    def test_cannot_enable_pipeline(self, fxt_pipeline, fxt_pipeline_service, fxt_client):
        # Mock activate_pipeline to raise ValidationError (which happens when update_pipeline raises it)
        fxt_pipeline_service.activate_pipeline.side_effect = ValidationError.from_exception_data(
            "Pipeline",
            [
                {
                    "type": "missing",
                    "loc": ("name",),
                    "msg": "Field required",
                    "input": {},
                },
            ],
        )

        response = fxt_client.post(f"/api/projects/{fxt_pipeline.project_id}/pipeline:run")

        assert response.status_code == status.HTTP_409_CONFLICT
        fxt_pipeline_service.activate_pipeline.assert_called_once_with(fxt_pipeline.project_id, set_running=True)

    def test_enable_pipeline_with_active_pipeline_different_project(
        self,
        fxt_pipeline,
        fxt_pipeline_service,
        fxt_client,
    ):
        """Test enabling a pipeline when another pipeline from a different project is already active."""
        other_project_id = uuid4()
        # Create a valid RUNNING pipeline with required fields
        source = VideoFileSourceConfig(
            id=uuid4(),
            project_id=other_project_id,
            source_type="video_file",
            name="Test Source",
            video_path="/path/to/video.mp4",
        )
        sink = FolderSinkConfig(
            id=uuid4(),
            project_id=other_project_id,
            sink_type="folder",
            name="Test Sink",
            folder_path="/path/to/output",
            output_formats=["image_original"],
            rate_limit=0.2,
        )
        model_id = uuid4()
        model = Model(
            id=model_id,
            project_id=other_project_id,
            name=f"Test Model ({str(model_id).split('-')[0]})",
            architecture="padim",
            format="openvino",
            train_job_id=uuid4(),
            dataset_snapshot_id=uuid4(),
        )
        active_pipeline = Pipeline(
            project_id=other_project_id,
            source=source,
            sink=sink,
            model=model,
            source_id=source.id,
            sink_id=sink.id,
            model_id=model.id,
            status=PipelineStatus.RUNNING,
        )
        # activate_pipeline raises ActivePipelineConflictError when there's an active pipeline from different project
        fxt_pipeline_service.activate_pipeline.side_effect = ActivePipelineConflictError(
            pipeline_id=str(fxt_pipeline.project_id),
            reason=f"another pipeline is already active. Please disable the pipeline {active_pipeline.id} "
            f"before activating a new one",
        )

        response = fxt_client.post(f"/api/projects/{fxt_pipeline.project_id}/pipeline:run")

        assert response.status_code == status.HTTP_409_CONFLICT
        assert "Cannot activate pipeline" in response.json()["detail"]
        fxt_pipeline_service.activate_pipeline.assert_called_once_with(fxt_pipeline.project_id, set_running=True)

    def test_enable_pipeline_with_active_pipeline_same_project(self, fxt_pipeline, fxt_pipeline_service, fxt_client):
        """Test enabling a pipeline when the same pipeline is already active (re-enabling)."""
        # Create a valid RUNNING pipeline with required fields
        source = VideoFileSourceConfig(
            id=uuid4(),
            project_id=fxt_pipeline.project_id,
            source_type="video_file",
            name="Test Source",
            video_path="/path/to/video.mp4",
        )
        sink = FolderSinkConfig(
            id=uuid4(),
            project_id=fxt_pipeline.project_id,
            sink_type="folder",
            name="Test Sink",
            folder_path="/path/to/output",
            output_formats=["image_original"],
            rate_limit=0.2,
        )
        model_id = uuid4()
        model = Model(
            id=model_id,
            project_id=fxt_pipeline.project_id,
            name=f"Test Model ({str(model_id).split('-')[0]})",
            architecture="padim",
            format="openvino",
            train_job_id=uuid4(),
            dataset_snapshot_id=uuid4(),
        )
        active_pipeline = Pipeline(
            project_id=fxt_pipeline.project_id,
            source=source,
            sink=sink,
            model=model,
            source_id=source.id,
            sink_id=sink.id,
            model_id=model_id,
            status=PipelineStatus.RUNNING,
        )
        # activate_pipeline internally calls get_active_pipeline and update_pipeline
        # Make activate_pipeline call update_pipeline when called

        async def activate_pipeline_side_effect(proj_id, set_running=False):
            await fxt_pipeline_service.update_pipeline(proj_id, {"status": PipelineStatus.RUNNING})
            return active_pipeline

        fxt_pipeline_service.get_active_pipeline = AsyncMock(return_value=active_pipeline)
        fxt_pipeline_service.update_pipeline = AsyncMock(return_value=active_pipeline)
        fxt_pipeline_service.activate_pipeline = AsyncMock(side_effect=activate_pipeline_side_effect)

        response = fxt_client.post(f"/api/projects/{fxt_pipeline.project_id}/pipeline:run")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_pipeline_service.activate_pipeline.assert_called_once_with(fxt_pipeline.project_id, set_running=True)
        # activate_pipeline internally calls update_pipeline when re-enabling
        fxt_pipeline_service.update_pipeline.assert_called_once_with(
            fxt_pipeline.project_id,
            {"status": PipelineStatus.RUNNING},
        )

    def test_get_pipeline_metrics_success(self, fxt_pipeline, fxt_pipeline_metrics_service, fxt_client):
        """Test successful retrieval of pipeline metrics with default time window."""
        mock_metrics = PipelineMetrics(
            time_window=TimeWindow(start=datetime.now(UTC), end=datetime.now(UTC), time_window=60),
            inference=InferenceMetrics(
                latency=LatencyMetrics(avg_ms=100.5, min_ms=50.0, max_ms=200.0, p95_ms=180.0, latest_ms=120.0),
                throughput=ThroughputMetrics(
                    avg_requests_per_second=30.0,
                    total_requests=1800,
                    max_requests_per_second=45.0,
                ),
            ),
        )
        fxt_pipeline_metrics_service.get_pipeline_metrics.return_value = mock_metrics

        response = fxt_client.get(f"/api/projects/{fxt_pipeline.project_id}/pipeline/metrics")

        assert response.status_code == status.HTTP_200_OK
        fxt_pipeline_metrics_service.get_pipeline_metrics.assert_called_once_with(fxt_pipeline.project_id, 60)

    def test_get_pipeline_metrics_invalid_pipeline_id(self, fxt_pipeline_metrics_service, fxt_client):
        """Test metrics endpoint with invalid pipeline ID format."""
        response = fxt_client.get("/api/projects/invalid-id/pipeline/metrics")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_pipeline_metrics_service.get_pipeline_metrics.assert_not_called()

    # Note: ResourceNotFoundError handling not implemented in current endpoints
    # def test_get_pipeline_metrics_pipeline_not_found(self, fxt_pipeline, fxt_pipeline_service, fxt_client):
    #     """Test metrics endpoint when pipeline doesn't exist."""
    #     fxt_pipeline_service.get_pipeline_metrics.side_effect = ResourceNotFoundError(
    #         ResourceType.PIPELINE, str(fxt_pipeline.project_id)
    #     )
    #     response = fxt_client.get(f"/api/projects/{fxt_pipeline.project_id}/pipeline/metrics")
    #     assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    #     fxt_pipeline_service.get_pipeline_metrics.assert_called_once_with(fxt_pipeline.project_id, 60)

    def test_get_pipeline_metrics_pipeline_not_running(self, fxt_pipeline, fxt_pipeline_metrics_service, fxt_client):
        """Test metrics endpoint when pipeline is not in running state."""
        fxt_pipeline_metrics_service.get_pipeline_metrics.side_effect = ValueError(
            "Cannot get metrics for a pipeline that is not running.",
        )

        response = fxt_client.get(f"/api/projects/{fxt_pipeline.project_id}/pipeline/metrics")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Cannot get metrics for a pipeline that is not running" in response.json()["detail"]
        fxt_pipeline_metrics_service.get_pipeline_metrics.assert_called_once_with(fxt_pipeline.project_id, 60)

    @pytest.mark.parametrize("invalid_time_window", [0, -1, 3601, 7200])
    def test_get_pipeline_metrics_invalid_time_window(
        self,
        invalid_time_window,
        fxt_pipeline,
        fxt_pipeline_metrics_service,
        fxt_client,
    ):
        """Test metrics endpoint with invalid time window values."""
        response = fxt_client.get(
            f"/api/projects/{fxt_pipeline.project_id}/pipeline/metrics?time_window={invalid_time_window}",
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Duration must be between 1 and 3600 seconds" in response.json()["detail"]
        fxt_pipeline_metrics_service.get_pipeline_metrics.assert_not_called()

    @pytest.mark.parametrize("valid_time_window", [1, 30, 300, 1800, 3600])
    def test_get_pipeline_metrics_valid_time_windows(
        self,
        valid_time_window,
        fxt_pipeline,
        fxt_pipeline_metrics_service,
        fxt_client,
    ):
        """Test metrics endpoint with various valid time window values."""
        mock_metrics = PipelineMetrics(
            time_window=TimeWindow(start=datetime.now(UTC), end=datetime.now(UTC), time_window=valid_time_window),
            inference=InferenceMetrics(
                latency=LatencyMetrics(avg_ms=100.0, min_ms=50.0, max_ms=200.0, p95_ms=180.0, latest_ms=120.0),
                throughput=ThroughputMetrics(
                    avg_requests_per_second=30.0,
                    total_requests=1800,
                    max_requests_per_second=45.0,
                ),
            ),
        )
        fxt_pipeline_metrics_service.get_pipeline_metrics.return_value = mock_metrics

        response = fxt_client.get(
            f"/api/projects/{fxt_pipeline.project_id}/pipeline/metrics?time_window={valid_time_window}",
        )

        assert response.status_code == status.HTTP_200_OK
        fxt_pipeline_metrics_service.get_pipeline_metrics.assert_called_once_with(
            fxt_pipeline.project_id,
            valid_time_window,
        )

    def test_get_pipeline_metrics_no_data_available(self, fxt_pipeline, fxt_pipeline_metrics_service, fxt_client):
        """Test metrics endpoint when no latency data is available."""
        mock_metrics = PipelineMetrics(
            time_window=TimeWindow(start=datetime.now(UTC), end=datetime.now(UTC), time_window=60),
            inference=InferenceMetrics(
                latency=LatencyMetrics(avg_ms=None, min_ms=None, max_ms=None, p95_ms=None, latest_ms=None),
                throughput=ThroughputMetrics(
                    avg_requests_per_second=None,
                    total_requests=None,
                    max_requests_per_second=None,
                ),
            ),
        )
        fxt_pipeline_metrics_service.get_pipeline_metrics.return_value = mock_metrics

        response = fxt_client.get(f"/api/projects/{fxt_pipeline.project_id}/pipeline/metrics")

        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()

        assert response_data["inference"]["latency"]["avg_ms"] is None
        assert response_data["inference"]["latency"]["min_ms"] is None
        assert response_data["inference"]["latency"]["max_ms"] is None
        assert response_data["inference"]["latency"]["p95_ms"] is None
        assert response_data["inference"]["latency"]["latest_ms"] is None

        fxt_pipeline_metrics_service.get_pipeline_metrics.assert_called_once_with(fxt_pipeline.project_id, 60)

    def test_get_pipeline_metrics_success_with_data(self, fxt_pipeline, fxt_pipeline_metrics_service, fxt_client):
        """Test successful retrieval of pipeline metrics with latency data."""
        mock_metrics = PipelineMetrics(
            time_window=TimeWindow(start=datetime.now(UTC), end=datetime.now(UTC), time_window=60),
            inference=InferenceMetrics(
                latency=LatencyMetrics(avg_ms=100.5, min_ms=50.0, max_ms=200.0, p95_ms=180.0, latest_ms=120.0),
                throughput=ThroughputMetrics(
                    avg_requests_per_second=30.0,
                    total_requests=1800,
                    max_requests_per_second=45.0,
                ),
            ),
        )
        fxt_pipeline_metrics_service.get_pipeline_metrics.return_value = mock_metrics

        response = fxt_client.get(f"/api/projects/{str(fxt_pipeline.project_id)}/pipeline/metrics")

        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()

        assert response_data["inference"]["latency"]["avg_ms"] == 100.5
        assert response_data["inference"]["latency"]["min_ms"] == 50.0
        assert response_data["inference"]["latency"]["max_ms"] == 200.0
        assert response_data["inference"]["latency"]["p95_ms"] == 180.0
        assert response_data["inference"]["latency"]["latest_ms"] == 120.0

        fxt_pipeline_metrics_service.get_pipeline_metrics.assert_called_once_with(fxt_pipeline.project_id, 60)
