# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch
from uuid import uuid4

import pytest
from pydantic import ValidationError

from pydantic_models import JobStatus
from pydantic_models.job import TrainJobPayload
from repositories.binary_repo import ImageBinaryRepository, ModelBinaryRepository
from services import TrainingService
from utils.callbacks import ProgressSyncParams


@pytest.fixture
def fxt_model_binary_repo():
    """Fixture for a mock model binary repository."""
    return MagicMock(spec=ModelBinaryRepository)


@pytest.fixture
def fxt_image_binary_repo():
    """Fixture for a mock image binary repository."""
    return MagicMock(spec=ImageBinaryRepository)


@pytest.fixture
def fxt_mock_job_service():
    """Fixture for a mock job service instance."""
    mock_job_service = MagicMock()
    mock_job_service.get_pending_train_job = AsyncMock()
    mock_job_service.update_job_status = AsyncMock()
    mock_job_service.get_job_by_id = AsyncMock()
    return mock_job_service


@pytest.fixture
def fxt_mock_model_service():
    """Fixture for a mock model service instance."""
    mock_model_service = MagicMock()
    mock_model_service.create_model = AsyncMock()
    mock_model_service.delete_model = AsyncMock()
    return mock_model_service


@pytest.fixture
def fxt_mock_binary_repos(fxt_model_binary_repo, fxt_image_binary_repo):
    """Fixture for mock binary repositories."""
    with (
        patch("services.training_service.ModelBinaryRepository") as mock_model_bin_repo_class,
        patch("services.dataset_snapshot_service.ImageBinaryRepository") as mock_image_bin_repo_class,
    ):
        mock_model_bin_repo_class.return_value = fxt_model_binary_repo
        mock_image_bin_repo_class.return_value = fxt_image_binary_repo
        yield mock_model_bin_repo_class, mock_image_bin_repo_class


@pytest.fixture
def fxt_mock_anomalib_components():
    """Fixture for mock Anomalib components."""
    with (
        patch("services.training_service.Folder") as mock_folder_class,
        patch("services.training_service.get_model") as mock_get_model,
        patch("services.training_service.Engine") as mock_engine_class,
    ):
        mock_folder = MagicMock()
        mock_folder_class.return_value = mock_folder

        mock_anomalib_model = MagicMock()
        mock_anomalib_model.name = "Padim"
        mock_get_model.return_value = mock_anomalib_model

        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_engine.export.return_value = "/path/to/exported/model"

        yield {
            "folder_class": mock_folder_class,
            "folder": mock_folder,
            "get_model": mock_get_model,
            "anomalib_model": mock_anomalib_model,
            "engine_class": mock_engine_class,
            "engine": mock_engine,
        }


@pytest.fixture
def fxt_mock_job_service_class(fxt_mock_job_service):
    """Fixture that patches JobService class."""
    with patch("services.training_service.JobService") as mock_job_service_class:
        mock_job_service_class.return_value = fxt_mock_job_service
        yield mock_job_service_class


@pytest.fixture
def fxt_mock_model_service_class(fxt_mock_model_service):
    """Fixture that patches ModelService class."""
    with patch("services.training_service.ModelService") as mock_model_service_class:
        mock_model_service_class.return_value = fxt_mock_model_service
        yield mock_model_service_class


@pytest.fixture
def fxt_mock_dataset_snapshot_service():
    """Fixture for mocking DatasetSnapshotService."""
    with patch("services.training_service.DatasetSnapshotService") as mock_service:
        # Setup default return values
        mock_snapshot = MagicMock()
        mock_snapshot.id = uuid4()
        mock_service.create_snapshot = AsyncMock(return_value=mock_snapshot)
        # Fix: get_or_create_snapshot needs to be awaited
        mock_service.get_or_create_snapshot = AsyncMock(return_value=mock_snapshot)
        mock_service.delete_snapshot_if_unused = AsyncMock()

        # Setup context manager for use_snapshot_as_folder
        mock_ctx = MagicMock()

        async def enter_mock(*args, **kwargs):
            return "/tmp/dataset"

        async def exit_mock(*args, **kwargs):
            return None

        mock_ctx.__aenter__ = AsyncMock(side_effect=enter_mock)
        mock_ctx.__aexit__ = AsyncMock(side_effect=exit_mock)
        mock_service.use_snapshot_as_folder.return_value = mock_ctx

        yield mock_service


class TestTrainingService:
    def test_train_pending_job_no_pending_jobs(self, fxt_mock_job_service_class, fxt_mock_job_service):
        """Test training when no pending jobs exist."""
        fxt_mock_job_service.get_pending_train_job.return_value = None

        result = asyncio.run(TrainingService.train_pending_job())

        assert result is None
        fxt_mock_job_service.get_pending_train_job.assert_called_once()

    def test_train_pending_job_success(
        self,
        fxt_job,
        fxt_model,
        fxt_mock_job_service_class,
        fxt_mock_model_service_class,
        fxt_mock_job_service,
        fxt_mock_model_service,
        fxt_mock_dataset_snapshot_service,
    ):
        """Test successful training of a pending job."""
        fxt_job.payload = {"model_name": "padim"}
        fxt_mock_job_service.get_pending_train_job.return_value = fxt_job
        fxt_mock_model_service.create_model.return_value = fxt_model

        with patch("services.training_service.asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = fxt_model

            result = asyncio.run(TrainingService.train_pending_job())

            assert result == fxt_model
            # Should be called twice: RUNNING then COMPLETED
            assert fxt_mock_job_service.update_job_status.call_count == 2
            fxt_mock_job_service.update_job_status.assert_any_call(
                job_id=fxt_job.id,
                status=JobStatus.RUNNING,
                message="Training started",
            )
            fxt_mock_job_service.update_job_status.assert_any_call(
                job_id=fxt_job.id,
                status=JobStatus.COMPLETED,
                message="Training completed successfully",
            )
            fxt_mock_model_service.create_model.assert_called_once()

            # Verify the persisted model carries the architecture identifier from the
            # payload and a name that embeds the generated model ID in the expected format.
            created_model = mock_to_thread.call_args.kwargs["model"]
            assert created_model.architecture == fxt_job.payload["model_name"]
            assert created_model.name == f"{fxt_job.payload['model_name']} ({str(created_model.id).split('-')[0]})"

    @pytest.mark.parametrize(
        "exception,expected_message",
        [
            (Exception("Training failed"), "Training failed"),
            (ValueError("Training failed - model is None"), "Training failed - model is None"),
        ],
    )
    def test_train_pending_job_training_failures(
        self,
        fxt_job,
        fxt_mock_job_service_class,
        fxt_mock_model_service_class,
        fxt_mock_job_service,
        fxt_mock_dataset_snapshot_service,
        exception,
        expected_message,
    ):
        """Test training failure handling with different failure scenarios."""
        fxt_job.payload = {"model_name": "padim"}
        fxt_mock_job_service.get_pending_train_job.return_value = fxt_job

        # Simulate that after failure, the job status in DB is FAILED (not active)
        failed_job = fxt_job.model_copy(update={"status": JobStatus.FAILED})
        fxt_mock_job_service.get_job_by_id.return_value = failed_job

        with patch("services.training_service.asyncio.to_thread") as mock_to_thread:
            if isinstance(exception, ValueError) and "model is None" in str(exception):
                mock_to_thread.return_value = None
            else:
                mock_to_thread.side_effect = exception

            with pytest.raises(type(exception), match=expected_message):
                asyncio.run(TrainingService.train_pending_job())

            # Should be called twice: RUNNING then FAILED
            assert fxt_mock_job_service.update_job_status.call_count == 2
            fxt_mock_job_service.update_job_status.assert_any_call(
                job_id=fxt_job.id,
                status=JobStatus.RUNNING,
                message="Training started",
            )
            fxt_mock_job_service.update_job_status.assert_any_call(
                job_id=fxt_job.id,
                status=JobStatus.FAILED,
                message=f"Failed with exception: {expected_message}",
            )

    def test_train_pending_job_cleanup_on_failure(
        self,
        fxt_job,
        fxt_model,
        fxt_mock_job_service_class,
        fxt_mock_model_service_class,
        fxt_mock_job_service,
        fxt_mock_model_service,
        fxt_mock_binary_repos,
        fxt_mock_dataset_snapshot_service,
    ):
        """Test cleanup when training fails and model has export_path."""
        fxt_job.payload = {"model_name": "padim"}
        fxt_mock_job_service.get_pending_train_job.return_value = fxt_job
        fxt_model.export_path = "/path/to/model"

        with patch("services.training_service.asyncio.to_thread") as mock_to_thread:
            # Mock the training to succeed first, setting export_path, then fail
            def mock_train_model(
                cls,
                model,
                synchronization_parameters: ProgressSyncParams,
                device_type=None,
                device_index=None,
                dataset_root=None,
                max_epochs=1,
            ):
                model.export_path = "/path/to/model"
                raise Exception("Training failed")

            mock_to_thread.side_effect = mock_train_model

            with pytest.raises(Exception, match="Training failed"):
                asyncio.run(TrainingService.train_pending_job())

            # Verify cleanup was called
            fxt_mock_model_service.delete_model.assert_called_once()
            call_args = fxt_mock_model_service.delete_model.call_args
            assert call_args[1]["project_id"] == fxt_job.project_id

    def test_train_model_success(
        self,
        fxt_model,
        fxt_model_binary_repo,
        fxt_image_binary_repo,
        fxt_mock_anomalib_components,
        fxt_mock_binary_repos,
    ):
        """Test successful model training with platform-specific worker configuration."""
        # Setup platform mock
        # Setup binary repo paths
        fxt_image_binary_repo.project_folder_path = "/path/to/images"
        fxt_model_binary_repo.model_folder_path = "/path/to/model"

        # Call the method
        with (
            patch.object(TrainingService, "_compute_export_size", return_value=123),
            patch.object(TrainingService, "_move_artifact") as mock_move_artifact,
            patch("services.training_service.Path.mkdir"),
        ):
            result = TrainingService._train_model(
                fxt_model,
                synchronization_parameters=ProgressSyncParams(),
                dataset_root="/tmp/dataset",
                max_epochs=42,
            )

        # Verify the result
        assert result == fxt_model
        assert fxt_model.is_ready is True
        assert fxt_model.export_path == "/path/to/model"
        assert fxt_model.size == 123

        # Verify all components were called correctly
        fxt_mock_anomalib_components["folder_class"].assert_called_once()
        fxt_mock_anomalib_components["get_model"].assert_called_once_with(model=fxt_model.architecture, evaluator=ANY)

        # Verify Engine was called with expected parameters
        fxt_mock_anomalib_components["engine_class"].assert_called_once()
        call_args = fxt_mock_anomalib_components["engine_class"].call_args
        assert call_args[1]["default_root_dir"] == Path("data") / f"train-workspace-{fxt_model.id}"
        assert "logger" in call_args[1]
        assert len(call_args[1]["logger"]) == 1  # tensorboard
        assert call_args[1]["max_epochs"] == 42

        fxt_mock_anomalib_components["engine"].fit.assert_called_once_with(
            model=fxt_mock_anomalib_components["anomalib_model"],
            datamodule=fxt_mock_anomalib_components["folder"],
        )
        fxt_mock_anomalib_components["engine"].export.assert_called_once()

        mock_move_artifact.assert_has_calls([
            call(
                src_dir=(
                    Path("data")
                    / f"train-workspace-{fxt_model.id}"
                    / "Padim"
                    / f"{fxt_model.project_id}-padim"
                    / "latest"
                    / "weights"
                    / "openvino"
                ).resolve(),
                dest_dir=Path("/path/to/model"),
                file_name="model.xml",
            ),
            call(
                src_dir=(
                    Path("data")
                    / f"train-workspace-{fxt_model.id}"
                    / "Padim"
                    / f"{fxt_model.project_id}-padim"
                    / "latest"
                    / "weights"
                    / "openvino"
                ).resolve(),
                dest_dir=Path("/path/to/model"),
                file_name="model.bin",
            ),
            call(
                src_dir=(
                    Path("data")
                    / f"train-workspace-{fxt_model.id}"
                    / "Padim"
                    / f"{fxt_model.project_id}-padim"
                    / "latest"
                    / "weights"
                    / "lightning"
                ).resolve(),
                dest_dir=Path("/path/to/model/checkpoint"),
                file_name="model.ckpt",
            ),
        ])

    def test_train_pending_job_cancelled(
        self,
        fxt_job,
        fxt_mock_job_service_class,
        fxt_mock_model_service_class,
        fxt_mock_job_service,
        fxt_mock_binary_repos,
        fxt_mock_dataset_snapshot_service,
    ):
        """Training should mark job as cancelled when cancellation flag is set."""
        fxt_job.payload = {"model_name": "padim"}
        fxt_mock_job_service.get_pending_train_job.return_value = fxt_job

        async def fake_train_model(*args: Any, **kwargs: Any) -> MagicMock:
            sync_params: ProgressSyncParams = kwargs["synchronization_parameters"]
            sync_params.set_cancel_training_event()
            return MagicMock()

        with (
            patch("services.training_service.asyncio.to_thread", side_effect=fake_train_model),
            patch("services.training_service.TrainingService._sync_progress_with_db", new=AsyncMock()),
        ):
            result = asyncio.run(TrainingService.train_pending_job())

        assert result is None
        fxt_mock_job_service.update_job_status.assert_any_call(
            job_id=fxt_job.id,
            status=JobStatus.RUNNING,
            message="Training started",
        )
        fxt_mock_job_service.update_job_status.assert_any_call(
            job_id=fxt_job.id,
            status=JobStatus.CANCELED,
            message="Training cancelled by user",
        )

    def test_train_model_cancelled_before_start(
        self,
        fxt_model,
        fxt_mock_anomalib_components,
        fxt_mock_binary_repos,
    ):
        """_train_model should abort early when cancellation is requested before training starts."""
        sync_params = ProgressSyncParams()
        sync_params.set_cancel_training_event()

        result = TrainingService._train_model(
            fxt_model,
            synchronization_parameters=sync_params,
            dataset_root="/tmp/dataset",
            max_epochs=1,
        )

        assert result is None

    @pytest.mark.parametrize(
        "invalid_model_name",
        [
            "x" * 65,  # exceeds Model.architecture max_length (64)
            "",  # empty
            "../x",  # path traversal attempt
            "foo/bar",  # path separator
            "foo bar",  # whitespace inside identifier
            "foo-bar",  # unsupported separator
        ],
    )
    def test_train_pending_job_invalid_model_name_fails_clearly(
        self,
        fxt_job,
        fxt_mock_job_service_class,
        fxt_mock_model_service_class,
        fxt_mock_job_service,
        fxt_mock_dataset_snapshot_service,
        invalid_model_name,
    ):
        """An invalid model_name must fail the job with a clear payload-validation message.

        The job should fail before any Model/training is constructed, so no snapshot or
        training work is started.
        """
        fxt_job.payload = {"model_name": invalid_model_name}
        fxt_mock_job_service.get_pending_train_job.return_value = fxt_job

        with patch("services.training_service.asyncio.to_thread") as mock_to_thread:
            result = asyncio.run(TrainingService.train_pending_job())

        # Job fails cleanly (returns None) without ever running training.
        assert result is None
        mock_to_thread.assert_not_called()
        fxt_mock_dataset_snapshot_service.get_or_create_snapshot.assert_not_called()

        # Status transitions: RUNNING then FAILED with a payload-validation message.
        fxt_mock_job_service.update_job_status.assert_any_call(
            job_id=fxt_job.id,
            status=JobStatus.RUNNING,
            message="Training started",
        )
        failed_calls = [
            call
            for call in fxt_mock_job_service.update_job_status.call_args_list
            if call.kwargs.get("status") == JobStatus.FAILED
        ]
        assert len(failed_calls) == 1
        assert "Failed to validate training job payload" in failed_calls[0].kwargs["message"]


class TestTrainJobPayloadValidation:
    """Validation of TrainJobPayload.model_name (length and safe identifier charset)."""

    def test_valid_model_name_is_normalized(self, fxt_project):
        """A valid model name is accepted, trimmed, and lowercased."""
        payload = TrainJobPayload.model_validate(
            {"project_id": str(fxt_project.id), "model_name": "  Padim  "},
        )
        assert payload.model_name == "padim"

    def test_model_name_exceeding_max_length_is_rejected(self, fxt_project):
        """A model name longer than the architecture max length raises a clear error."""
        with pytest.raises(ValidationError, match="at most 64 characters"):
            TrainJobPayload.model_validate(
                {"project_id": str(fxt_project.id), "model_name": "x" * 65},
            )

    def test_empty_model_name_is_rejected(self, fxt_project):
        """An empty/whitespace-only model name raises a clear error."""
        with pytest.raises(ValidationError, match="must not be empty"):
            TrainJobPayload.model_validate(
                {"project_id": str(fxt_project.id), "model_name": "   "},
            )

    @pytest.mark.parametrize(
        "unsafe_model_name",
        [
            "../x",  # path traversal attempt
            "foo/bar",  # path separator
            "foo\\bar",  # windows path separator
            "foo bar",  # whitespace inside identifier
            "foo-bar",  # unsupported separator
            "foo.bar",  # dot (extension-like)
            "foo$",  # special character
        ],
    )
    def test_unsafe_model_name_is_rejected(self, fxt_project, unsafe_model_name):
        """Names containing path separators or unsafe characters raise a clear error."""
        with pytest.raises(ValidationError, match=r"\[a-z0-9_\]\+"):
            TrainJobPayload.model_validate(
                {"project_id": str(fxt_project.id), "model_name": unsafe_model_name},
            )

    def test_uppercase_model_name_is_lowercased(self, fxt_project):
        """Mixed-case names are normalized to lowercase rather than rejected."""
        payload = TrainJobPayload.model_validate(
            {"project_id": str(fxt_project.id), "model_name": "EfficientAd"},
        )
        assert payload.model_name == "efficientad"
