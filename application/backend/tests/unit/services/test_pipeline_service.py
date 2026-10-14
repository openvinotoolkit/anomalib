# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
from multiprocessing.synchronize import Condition
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from pydantic_models import Pipeline, PipelineStatus
from pydantic_models.model import Model
from pydantic_models.sink import FolderSinkConfig, MqttSinkConfig
from pydantic_models.source import UsbCameraSourceConfig, VideoFileSourceConfig
from repositories import PipelineRepository
from services import ActivePipelineService, ModelService, ResourceNotFoundError
from services.metrics_service import MetricsService
from services.pipeline_service import PipelineService


@pytest.fixture
def fxt_pipeline(fxt_project, fxt_model):
    """Fixture for a test pipeline."""
    source = VideoFileSourceConfig(
        id=uuid4(),
        project_id=fxt_project.id,
        source_type="video_file",
        name="Test Source",
        video_path="/path/to/video.mp4",
    )
    sink = FolderSinkConfig(
        id=uuid4(),
        project_id=fxt_project.id,
        sink_type="folder",
        name="Test Sink",
        folder_path="/path/to/output",
        output_formats=["image_original"],
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
def fxt_idle_pipeline(fxt_project):
    """Fixture for an idle pipeline."""
    return Pipeline(project_id=fxt_project.id, status=PipelineStatus.IDLE)


@pytest.fixture
def fxt_condition():
    """Fixture for a mock condition."""
    return MagicMock(spec=Condition)


@pytest.fixture
def fxt_active_pipeline_service():
    """Fixture for a mock active pipeline service."""
    service = MagicMock(spec=ActivePipelineService)
    service.reload = AsyncMock()
    return service


@pytest.fixture
def fxt_metrics_service():
    """Fixture for a mock metrics service."""
    service = MagicMock(spec=MetricsService)
    service.get_latency_measurements.return_value = [10.5, 12.3, 8.7, 15.2, 9.1]
    return service


@pytest.fixture
def fxt_model_service():
    """Fixture for a mock model service."""
    service = MagicMock(spec=ModelService)
    service.activate_model = MagicMock()
    return service


@pytest.fixture
def fxt_pipeline_service(fxt_active_pipeline_service, fxt_condition, fxt_model_service):
    """Fixture for PipelineService with mocked dependencies."""
    return PipelineService(
        active_pipeline_service=fxt_active_pipeline_service,
        config_changed_condition=fxt_condition,
        model_service=fxt_model_service,
    )


@pytest.fixture
def fxt_pipeline_repository():
    """Fixture for a mock pipeline repository."""
    return MagicMock(spec=PipelineRepository)


@pytest.fixture(autouse=True)
def mock_db_context():
    """Mock the database context for all tests."""
    with patch("services.pipeline_service.get_async_db_session_ctx") as mock_db_ctx:
        mock_session = AsyncMock()
        mock_db_ctx.return_value.__aenter__.return_value = mock_session
        mock_db_ctx.return_value.__aexit__.return_value = None
        yield mock_db_ctx


class TestPipelineService:
    def test_init(self, fxt_active_pipeline_service, fxt_condition, fxt_model_service):
        """Test PipelineService initialization."""
        service = PipelineService(
            active_pipeline_service=fxt_active_pipeline_service,
            config_changed_condition=fxt_condition,
            model_service=fxt_model_service,
        )

        assert service._active_pipeline_service == fxt_active_pipeline_service
        assert service._config_changed_condition == fxt_condition
        assert service._model_service == fxt_model_service

    def test_notify_source_changed(self, fxt_pipeline_service, fxt_condition):
        """Test source change notification."""
        fxt_pipeline_service._notify_source_changed()

        fxt_condition.__enter__.assert_called_once()
        fxt_condition.__exit__.assert_called_once()
        fxt_condition.notify_all.assert_called_once()

    def test_notify_sink_changed(self, fxt_pipeline_service, fxt_active_pipeline_service):
        """Test sink change notification."""
        asyncio.run(fxt_pipeline_service._notify_sink_changed())

        fxt_active_pipeline_service.reload.assert_called_once()

    def test_notify_pipeline_changed(self, fxt_pipeline_service, fxt_condition, fxt_active_pipeline_service):
        """Test pipeline change notification."""
        asyncio.run(fxt_pipeline_service._notify_pipeline_changed())

        fxt_condition.__enter__.assert_called_once()
        fxt_condition.__exit__.assert_called_once()
        fxt_condition.notify_all.assert_called_once()
        fxt_active_pipeline_service.reload.assert_called_once()

    @pytest.mark.parametrize(
        "has_session,session_param,expected_result,should_raise",
        [
            (False, None, "pipeline_found", False),
            (True, "fxt_db_session", "pipeline_found", False),
            (False, None, None, True),
        ],
    )
    def test_get_pipeline_by_id(
        self,
        fxt_pipeline,
        fxt_pipeline_repository,
        fxt_db_session,
        has_session,
        session_param,
        expected_result,
        should_raise,
    ):
        """Test getting pipeline by ID with different scenarios."""
        if should_raise:
            fxt_pipeline_repository.get_by_id.return_value = None
            project_id = uuid4()
        else:
            fxt_pipeline_repository.get_by_id.return_value = fxt_pipeline
            project_id = fxt_pipeline.project_id

        with patch("services.pipeline_service.PipelineRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_pipeline_repository

            if should_raise:
                with pytest.raises(ResourceNotFoundError) as exc_info:
                    asyncio.run(PipelineService.get_pipeline_by_id(project_id))
                assert str(project_id) in str(exc_info.value)
                assert "pipeline" in str(exc_info.value).lower()
            else:
                if has_session:
                    result = asyncio.run(PipelineService.get_pipeline_by_id(project_id, fxt_db_session))
                else:
                    result = asyncio.run(PipelineService.get_pipeline_by_id(project_id))
                assert result == fxt_pipeline

        fxt_pipeline_repository.get_by_id.assert_called_once_with(project_id)

    def test_update_pipeline_success(self, fxt_pipeline_service, fxt_pipeline, fxt_pipeline_repository):
        """Test updating pipeline successfully."""
        updated_pipeline = fxt_pipeline.model_copy(update={"status": PipelineStatus.IDLE})
        fxt_pipeline_repository.get_by_id.return_value = fxt_pipeline
        fxt_pipeline_repository.update.return_value = updated_pipeline

        with patch("services.pipeline_service.PipelineRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_pipeline_repository

            result = asyncio.run(
                fxt_pipeline_service.update_pipeline(fxt_pipeline.project_id, {"status": PipelineStatus.IDLE}),
            )

        assert result == updated_pipeline
        fxt_pipeline_repository.get_by_id.assert_called_once_with(fxt_pipeline.project_id)
        fxt_pipeline_repository.update.assert_called_once_with(fxt_pipeline, {"status": PipelineStatus.IDLE})

    def test_update_pipeline_running_to_running_source_change(
        self,
        fxt_pipeline_service,
        fxt_pipeline,
        fxt_pipeline_repository,
        fxt_condition,
    ):
        """Test updating running pipeline with source change."""
        new_source = UsbCameraSourceConfig(
            id=uuid4(),
            project_id=fxt_pipeline.project_id,
            source_type="usb_camera",
            name="New Source",
            device_id=0,
        )
        updated_pipeline = fxt_pipeline.model_copy(update={"source": new_source, "source_id": new_source.id})
        fxt_pipeline_repository.get_by_id.return_value = fxt_pipeline
        fxt_pipeline_repository.update.return_value = updated_pipeline

        with patch("services.pipeline_service.PipelineRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_pipeline_repository

            result = asyncio.run(
                fxt_pipeline_service.update_pipeline(
                    fxt_pipeline.project_id,
                    {"source": new_source, "source_id": new_source.id},
                ),
            )

        assert result == updated_pipeline
        fxt_condition.notify_all.assert_called()

    def test_update_pipeline_running_to_running_sink_change(
        self,
        fxt_pipeline_service,
        fxt_pipeline,
        fxt_pipeline_repository,
        fxt_active_pipeline_service,
    ):
        """Test updating running pipeline with sink change."""
        new_sink = MqttSinkConfig(
            id=uuid4(),
            project_id=fxt_pipeline.project_id,
            sink_type="mqtt",
            name="New Sink",
            broker_host="localhost",
            broker_port=1883,
            topic="anomaly_detection",
            output_formats=["predictions"],
            rate_limit=0.1,
        )
        updated_pipeline = fxt_pipeline.model_copy(update={"sink": new_sink, "sink_id": new_sink.id})
        fxt_pipeline_repository.get_by_id.return_value = fxt_pipeline
        fxt_pipeline_repository.update.return_value = updated_pipeline

        with patch("services.pipeline_service.PipelineRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_pipeline_repository

            result = asyncio.run(
                fxt_pipeline_service.update_pipeline(
                    fxt_pipeline.project_id,
                    {"sink": new_sink, "sink_id": new_sink.id},
                ),
            )

        assert result == updated_pipeline
        fxt_active_pipeline_service.reload.assert_called_once()

    def test_update_pipeline_running_to_running_model_change(
        self,
        fxt_pipeline_service,
        fxt_pipeline,
        fxt_pipeline_repository,
        fxt_model_service,
    ):
        """Test updating running pipeline with model change."""
        model_id = uuid4()
        new_model = Model(
            id=model_id,
            project_id=fxt_pipeline.project_id,
            name=f"new_model ({str(model_id).split('-')[0]})",
            architecture="padim",
            format="openvino",
            train_job_id=uuid4(),
            dataset_snapshot_id=uuid4(),
        )
        updated_pipeline = fxt_pipeline.model_copy(update={"model": new_model, "model_id": model_id})
        fxt_pipeline_repository.get_by_id.return_value = fxt_pipeline
        fxt_pipeline_repository.update.return_value = updated_pipeline

        with patch("services.pipeline_service.PipelineRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_pipeline_repository

            result = asyncio.run(
                fxt_pipeline_service.update_pipeline(
                    fxt_pipeline.project_id,
                    {"model": new_model, "model_id": model_id},
                ),
            )

        assert result == updated_pipeline
        fxt_model_service.activate_model.assert_called_once()

    def test_update_pipeline_status_change_to_running(
        self,
        fxt_pipeline_service,
        fxt_idle_pipeline,
        fxt_pipeline_repository,
        fxt_condition,
        fxt_active_pipeline_service,
        fxt_model_service,
    ):
        """Test updating pipeline status from idle to running."""
        updated_pipeline = fxt_idle_pipeline.model_copy(update={"status": PipelineStatus.RUNNING})
        fxt_pipeline_repository.get_by_id.return_value = fxt_idle_pipeline
        fxt_pipeline_repository.update.return_value = updated_pipeline

        with patch("services.pipeline_service.PipelineRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_pipeline_repository

            result = asyncio.run(
                fxt_pipeline_service.update_pipeline(fxt_idle_pipeline.project_id, {"status": PipelineStatus.RUNNING}),
            )

        assert result == updated_pipeline
        fxt_condition.__enter__.assert_called_once()
        fxt_condition.__exit__.assert_called_once()
        fxt_condition.notify_all.assert_called_once()
        fxt_active_pipeline_service.reload.assert_called_once()
        # activate_model is called on status change regardless of whether model exists
        fxt_model_service.activate_model.assert_called_once()

    def test_update_pipeline_status_change_to_running_with_model(
        self,
        fxt_pipeline_service,
        fxt_idle_pipeline,
        fxt_model,
        fxt_pipeline_repository,
        fxt_condition,
        fxt_active_pipeline_service,
        fxt_model_service,
    ):
        """Test updating pipeline status to running with model."""
        updated_pipeline = fxt_idle_pipeline.model_copy(update={"status": PipelineStatus.RUNNING, "model": fxt_model})
        fxt_pipeline_repository.get_by_id.return_value = fxt_idle_pipeline
        fxt_pipeline_repository.update.return_value = updated_pipeline

        with patch("services.pipeline_service.PipelineRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_pipeline_repository

            result = asyncio.run(
                fxt_pipeline_service.update_pipeline(
                    fxt_idle_pipeline.project_id,
                    {"status": PipelineStatus.RUNNING, "model": fxt_model},
                ),
            )

        assert result == updated_pipeline
        fxt_model_service.activate_model.assert_called_once()

    def test_get_active_pipeline(self, fxt_pipeline, fxt_pipeline_repository):
        """Test getting the active pipeline from the database."""
        fxt_pipeline_repository.get_active_pipeline.return_value = fxt_pipeline

        with patch("services.pipeline_service.PipelineRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_pipeline_repository

            result = asyncio.run(PipelineService.get_active_pipeline())

        assert result == fxt_pipeline
        fxt_pipeline_repository.get_active_pipeline.assert_called_once()

    def test_get_active_pipeline_none(self, fxt_pipeline_repository):
        """Test getting active pipeline when no pipeline is running."""
        fxt_pipeline_repository.get_active_pipeline.return_value = None

        with patch("services.pipeline_service.PipelineRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_pipeline_repository

            result = asyncio.run(PipelineService.get_active_pipeline())

        assert result is None
        fxt_pipeline_repository.get_active_pipeline.assert_called_once()

    def test_get_active_pipeline_returns_running_pipeline(self, fxt_pipeline, fxt_pipeline_repository):
        """Test that get_active_pipeline returns a pipeline with RUNNING status and required fields."""
        fxt_pipeline_repository.get_active_pipeline.return_value = fxt_pipeline

        with patch("services.pipeline_service.PipelineRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_pipeline_repository

            result = asyncio.run(PipelineService.get_active_pipeline())

        assert result is not None
        assert result.status == PipelineStatus.RUNNING
        assert result.source is not None
        assert result.sink is not None
        assert result.model is not None
        assert result == fxt_pipeline
        fxt_pipeline_repository.get_active_pipeline.assert_called_once()
