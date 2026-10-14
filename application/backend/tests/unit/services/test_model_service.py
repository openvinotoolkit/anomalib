# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np
import openvino.properties.hint as ov_hints
import pytest
from anomalib.deploy import OpenVINOInferencer

from pydantic_models import PredictionLabel
from repositories import ModelRepository
from services import ModelService


@pytest.fixture
def fxt_model_repository():
    """Fixture for a mock model repository."""
    return MagicMock(spec=ModelRepository)


@pytest.fixture
def fxt_model_service():
    """Fixture for ModelService - most methods are static, predict_image is instance method."""
    return ModelService


@pytest.fixture
def fxt_mp_event():
    """Fixture for a mock multiprocessing event."""
    return MagicMock()


@pytest.fixture(autouse=True)
def mock_db_context():
    """Mock the database context for all tests."""
    with patch("services.model_service.get_async_db_session_ctx") as mock_db_ctx:
        mock_session = AsyncMock()
        mock_db_ctx.return_value.__aenter__.return_value = mock_session
        mock_db_ctx.return_value.__aexit__.return_value = None
        yield mock_db_ctx


class TestModelService:
    def test_init_without_event(self, fxt_model_service):
        """Test ModelService initialization without multiprocessing event."""
        service = fxt_model_service()
        assert service._mp_model_reload_event is None

    def test_init_with_event(self, fxt_model_service, fxt_mp_event):
        """Test ModelService initialization with multiprocessing event."""
        service = fxt_model_service(mp_model_reload_event=fxt_mp_event)
        assert service._mp_model_reload_event == fxt_mp_event

    def test_activate_model_with_event(self, fxt_mp_event):
        """Test activate_model with multiprocessing event set."""
        service = ModelService(mp_model_reload_event=fxt_mp_event)
        service.activate_model()
        fxt_mp_event.set.assert_called_once()

    def test_activate_model_without_event(self):
        """Test activate_model without multiprocessing event (should not raise)."""
        service = ModelService()
        # Should not raise any exception
        service.activate_model()

    def test_activate_model_event_exception(self, fxt_mp_event):
        """Test activate_model when event.set() raises an exception."""
        fxt_mp_event.set.side_effect = Exception("Test exception")
        service = ModelService(mp_model_reload_event=fxt_mp_event)

        # Should not raise the exception, just log it
        service.activate_model()
        fxt_mp_event.set.assert_called_once()

    def test_create_model(self, fxt_model_service, fxt_model_repository, fxt_model):
        """Test creating a model."""
        fxt_model_repository.save.return_value = fxt_model

        with patch("services.model_service.ModelRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_model_repository

            result = asyncio.run(fxt_model_service.create_model(fxt_model))

        assert result == fxt_model
        fxt_model_repository.save.assert_called_once_with(fxt_model)

    def test_get_model_list(self, fxt_model_service, fxt_model_repository, fxt_model_list, fxt_project):
        """Test getting model list."""
        fxt_model_repository.get_all_count = AsyncMock(return_value=len(fxt_model_list.models))
        fxt_model_repository.get_all_pagination = AsyncMock(return_value=fxt_model_list.models)

        with patch("services.model_service.ModelRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_model_repository

            result = asyncio.run(fxt_model_service.get_model_list(fxt_project.id, limit=20, offset=0))

        assert result.models == fxt_model_list.models
        assert result.pagination.total == len(fxt_model_list.models)
        fxt_model_repository.get_all_count.assert_called_once()
        fxt_model_repository.get_all_pagination.assert_called_once_with(limit=20, offset=0)

    def test_get_model_by_id(self, fxt_model_service, fxt_model_repository, fxt_model, fxt_project):
        """Test getting model by ID."""
        fxt_model_repository.get_by_id.return_value = fxt_model

        with patch("services.model_service.ModelRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_model_repository

            result = asyncio.run(fxt_model_service.get_model_by_id(fxt_project.id, fxt_model.id))

        assert result == fxt_model
        fxt_model_repository.get_by_id.assert_called_once_with(fxt_model.id)

    def test_get_model_by_id_not_found(self, fxt_model_service, fxt_model_repository, fxt_project):
        """Test getting model by ID when not found."""
        fxt_model_repository.get_by_id.return_value = None

        with patch("services.model_service.ModelRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_model_repository

            result = asyncio.run(fxt_model_service.get_model_by_id(fxt_project.id, "non-existent-id"))

        assert result is None
        fxt_model_repository.get_by_id.assert_called_once_with("non-existent-id")

    def test_delete_model(self, fxt_model_service, fxt_model_repository, fxt_model, fxt_project):
        """Test deleting a model."""
        fxt_model_repository.delete_by_id.return_value = None
        fxt_model_repository.get_by_id.return_value = fxt_model

        with (
            patch("services.model_service.ModelRepository") as mock_repo_class,
            patch("services.model_service.DatasetSnapshotService") as mock_snapshot_service,
            patch("services.model_service.ModelBinaryRepository") as mock_binary_repo_class,
        ):
            mock_repo_class.return_value = fxt_model_repository

            mock_binary_repo = MagicMock()
            mock_binary_repo_class.return_value = mock_binary_repo
            mock_binary_repo.delete_model_folder = AsyncMock()

            # Mock async method
            mock_snapshot_service.delete_snapshot_if_unused = AsyncMock()

            asyncio.run(fxt_model_service.delete_model(fxt_project.id, fxt_model.id))

        fxt_model_repository.delete_by_id.assert_called_once_with(fxt_model.id)
        mock_snapshot_service.delete_snapshot_if_unused.assert_called_once_with(
            snapshot_id=fxt_model.dataset_snapshot_id,
            project_id=fxt_project.id,
        )
        mock_binary_repo.delete_model_folder.assert_called_once()

    def test_delete_model_also_deletes_training_job(
        self,
        fxt_model_service,
        fxt_model_repository,
        fxt_model,
        fxt_project,
    ):
        """Test that deleting a model also deletes its associated training job."""
        fxt_model_repository.delete_by_id.return_value = None
        fxt_model_repository.get_by_id.return_value = fxt_model
        fxt_model.train_job_id = uuid4()

        with (
            patch("services.model_service.ModelRepository") as mock_repo_class,
            patch("services.model_service.JobRepository") as mock_job_repo_class,
            patch("services.model_service.DatasetSnapshotService") as mock_snapshot_service,
            patch("services.model_service.ModelBinaryRepository") as mock_binary_repo_class,
        ):
            mock_repo_class.return_value = fxt_model_repository
            mock_job_repo = MagicMock()
            mock_job_repo.delete_by_id = AsyncMock()
            mock_job_repo_class.return_value = mock_job_repo
            mock_binary_repo_class.return_value.delete_model_folder = AsyncMock()
            mock_snapshot_service.delete_snapshot_if_unused = AsyncMock()

            asyncio.run(fxt_model_service.delete_model(fxt_project.id, fxt_model.id))

        mock_job_repo.delete_by_id.assert_called_once_with(fxt_model.train_job_id)

    def test_load_inference_model_success(self, fxt_model_service, fxt_model, fxt_openvino_inferencer):
        """Test loading inference model successfully."""
        with patch("services.model_service.ModelBinaryRepository") as mock_bin_repo_class:
            mock_bin_repo = MagicMock()
            mock_bin_repo_class.return_value = mock_bin_repo
            mock_bin_repo.get_weights_file_path.return_value = "/path/to/model.xml"

            with patch("services.model_service.asyncio.to_thread") as mock_to_thread:
                mock_to_thread.return_value = fxt_openvino_inferencer

                result = asyncio.run(fxt_model_service.load_inference_model(fxt_model, "CPU"))

                assert result == fxt_openvino_inferencer
                mock_bin_repo.get_weights_file_path.assert_called_once_with(
                    name="model.xml",
                )
                mock_to_thread.assert_called_once_with(
                    OpenVINOInferencer,
                    path="/path/to/model.xml",
                    device="CPU",
                    config={ov_hints.performance_mode: ov_hints.PerformanceMode.LATENCY},
                )

    def test_load_inference_model_unsupported_format(self, fxt_model_service, fxt_model):
        """Test loading inference model with unsupported format."""
        fxt_model.format = "unsupported_format"

        with pytest.raises(NotImplementedError) as exc_info:
            asyncio.run(fxt_model_service.load_inference_model(fxt_model))

        assert "Model format unsupported_format is not supported" in str(exc_info.value)

    @pytest.mark.parametrize("device", ["CPU", "GPU", "AUTO"])
    def test_load_inference_model_with_different_devices(
        self,
        fxt_model_service,
        fxt_model,
        fxt_openvino_inferencer,
        device,
    ):
        """Test loading inference model with different devices."""
        with patch("services.model_service.ModelBinaryRepository") as mock_bin_repo_class:
            mock_bin_repo = MagicMock()
            mock_bin_repo_class.return_value = mock_bin_repo
            mock_bin_repo.get_weights_file_path.return_value = "/path/to/model.xml"

            with patch("services.model_service.asyncio.to_thread") as mock_to_thread:
                mock_to_thread.return_value = fxt_openvino_inferencer

                result = asyncio.run(fxt_model_service.load_inference_model(fxt_model, device))

                assert result == fxt_openvino_inferencer
                mock_to_thread.assert_called_once_with(
                    OpenVINOInferencer,
                    path="/path/to/model.xml",
                    device=device,
                    config={ov_hints.performance_mode: ov_hints.PerformanceMode.LATENCY},
                )

    def test_predict_image_with_cached_model(
        self,
        fxt_model_service,
        fxt_model,
        fxt_image_bytes,
        fxt_prediction_response,
        fxt_openvino_inferencer,
    ):
        """Test prediction with cached model."""
        cached_models = {fxt_model.id: fxt_openvino_inferencer}

        with patch.object(fxt_model_service, "_run_prediction_pipeline") as mock_pipeline:
            mock_pipeline.return_value = {
                "anomaly_map": "base64_encoded_image",
                "label": PredictionLabel.NORMAL,
                "score": 0.1,
            }

            result = asyncio.run(fxt_model_service().predict_image(fxt_model, fxt_image_bytes, cached_models))

            assert result.anomaly_map == "base64_encoded_image"
            assert result.label == PredictionLabel.NORMAL
            assert result.score == 0.1
            mock_pipeline.assert_called_once()

    def test_predict_image_without_cached_model(
        self,
        fxt_model_service,
        fxt_model,
        fxt_image_bytes,
        fxt_openvino_inferencer,
    ):
        """Test prediction without cached model."""
        with patch.object(fxt_model_service, "load_inference_model") as mock_load_model:
            mock_load_model.return_value = fxt_openvino_inferencer

            with patch.object(fxt_model_service, "_run_prediction_pipeline") as mock_pipeline:
                mock_pipeline.return_value = {
                    "anomaly_map": "base64_encoded_image",
                    "label": PredictionLabel.NORMAL,
                    "score": 0.1,
                }

                result = asyncio.run(fxt_model_service().predict_image(fxt_model, fxt_image_bytes, {}))

                assert result.anomaly_map == "base64_encoded_image"
                assert result.label == PredictionLabel.NORMAL
                assert result.score == 0.1
                mock_load_model.assert_called_once_with(fxt_model, device=None)
                mock_pipeline.assert_called_once()

    def test_predict_image_caches_model(self, fxt_model_service, fxt_model, fxt_image_bytes, fxt_openvino_inferencer):
        """Test that prediction caches the loaded model."""
        cached_models = {}

        with patch.object(fxt_model_service, "load_inference_model") as mock_load_model:
            mock_load_model.return_value = fxt_openvino_inferencer

            with patch.object(fxt_model_service, "_run_prediction_pipeline") as mock_pipeline:
                mock_pipeline.return_value = {
                    "anomaly_map": "base64_encoded_image",
                    "label": PredictionLabel.NORMAL,
                    "score": 0.1,
                }

                asyncio.run(fxt_model_service().predict_image(fxt_model, fxt_image_bytes, cached_models))

                assert fxt_model.id in cached_models
                assert cached_models[fxt_model.id] == fxt_openvino_inferencer

    def test_predict_image_load_model_exception(self, fxt_model_service, fxt_model, fxt_image_bytes):
        """Test prediction when load_inference_model raises an exception."""
        with patch.object(fxt_model_service, "load_inference_model") as mock_load_model:
            mock_load_model.side_effect = Exception("Model loading failed")

            with pytest.raises(Exception) as exc_info:
                asyncio.run(fxt_model_service().predict_image(fxt_model, fxt_image_bytes, {}))

            assert "Model loading failed" in str(exc_info.value)
            mock_load_model.assert_called_once_with(fxt_model, device=None)

    @patch("services.model_service.cv2.imdecode")
    @patch("services.model_service.cv2.cvtColor")
    def test_run_prediction_pipeline(self, mock_cvt_color, mock_imdecode, fxt_openvino_inferencer, fxt_image_bytes):
        """Test the prediction pipeline static method."""
        # Mock OpenCV functions to avoid actual image processing
        mock_imdecode.return_value = np.array(
            [[[100, 100, 100], [200, 200, 200]], [[150, 150, 150], [250, 250, 250]]],
            dtype=np.uint8,
        )
        mock_cvt_color.return_value = np.array(
            [[[100, 100, 100, 255], [200, 200, 200, 255]], [[150, 150, 150, 255], [250, 250, 250, 255]]],
            dtype=np.uint8,
        )

        # Create a test anomaly map
        test_anomaly_map = np.array([[[0.1, 0.2], [0.3, 0.4]]])
        fxt_openvino_inferencer.predict.return_value.anomaly_map = test_anomaly_map
        fxt_openvino_inferencer.predict.return_value.pred_label.item.return_value = 0
        fxt_openvino_inferencer.predict.return_value.pred_score.item.return_value = 0.1

        result = ModelService._run_prediction_pipeline(fxt_openvino_inferencer, fxt_image_bytes)

        assert "anomaly_map" in result
        assert result["label"] == PredictionLabel.NORMAL
        assert result["score"] == 0.1
        assert isinstance(result["anomaly_map"], str)  # Should be base64 encoded

    @patch("services.model_service.cv2.imdecode")
    @patch("services.model_service.cv2.cvtColor")
    def test_run_prediction_pipeline_anomalous(
        self,
        mock_cvt_color,
        mock_imdecode,
        fxt_openvino_inferencer,
        fxt_image_bytes,
    ):
        """Test the prediction pipeline with anomalous prediction."""
        # Mock OpenCV functions to avoid actual image processing
        mock_imdecode.return_value = np.array(
            [[[100, 100, 100], [200, 200, 200]], [[150, 150, 150], [250, 250, 250]]],
            dtype=np.uint8,
        )
        mock_cvt_color.return_value = np.array(
            [[[100, 100, 100, 255], [200, 200, 200, 255]], [[150, 150, 150, 255], [250, 250, 250, 255]]],
            dtype=np.uint8,
        )

        # Create a test anomaly map
        test_anomaly_map = np.array([[[0.8, 0.9], [0.7, 0.6]]])
        fxt_openvino_inferencer.predict.return_value.anomaly_map = test_anomaly_map
        fxt_openvino_inferencer.predict.return_value.pred_label.item.return_value = 1
        fxt_openvino_inferencer.predict.return_value.pred_score.item.return_value = 0.9

        result = ModelService._run_prediction_pipeline(fxt_openvino_inferencer, fxt_image_bytes)

        assert "anomaly_map" in result
        assert result["label"] == PredictionLabel.ANOMALOUS
        assert result["score"] == 0.9
        assert isinstance(result["anomaly_map"], str)  # Should be base64 encoded

    @patch("services.model_service.cv2.imdecode")
    @patch("services.model_service.cv2.cvtColor")
    def test_run_prediction_pipeline_edge_cases(self, mock_cvt_color, mock_imdecode, fxt_openvino_inferencer):
        """Test prediction pipeline with edge cases."""
        # Test with empty image bytes
        empty_bytes = b""
        mock_imdecode.return_value = None  # cv2.imdecode returns None for invalid data
        mock_cvt_color.return_value = None

        # This should handle the case where imdecode returns None
        with pytest.raises(ValueError, match="Failed to decode image"):
            ModelService._run_prediction_pipeline(fxt_openvino_inferencer, empty_bytes)

    @patch("services.model_service.cv2.imdecode")
    @patch("services.model_service.cv2.cvtColor")
    def test_run_prediction_pipeline_advanced_processing(self, mock_cvt_color, mock_imdecode, fxt_openvino_inferencer):
        """Test prediction pipeline with advanced processing scenarios."""
        # Mock OpenCV functions
        mock_imdecode.return_value = np.array(
            [[[100, 100, 100], [200, 200, 200]], [[150, 150, 150], [250, 250, 250]]],
            dtype=np.uint8,
        )
        mock_cvt_color.return_value = np.array(
            [[[100, 100, 100, 255], [200, 200, 200, 255]], [[150, 150, 150, 255], [250, 250, 250, 255]]],
            dtype=np.uint8,
        )

        # Test with 4D anomaly map (should be squeezed to 2D)
        test_anomaly_map = np.array([[[[0.1, 0.2], [0.3, 0.4]]]])  # 4D array
        fxt_openvino_inferencer.predict.return_value.anomaly_map = test_anomaly_map
        fxt_openvino_inferencer.predict.return_value.pred_label.item.return_value = 0
        fxt_openvino_inferencer.predict.return_value.pred_score.item.return_value = 0.123456789

        result = ModelService._run_prediction_pipeline(fxt_openvino_inferencer, b"test_image")

        # Verify score precision handling
        assert result["score"] == 0.123456789
        assert isinstance(result["score"], float)

        # Verify anomaly map processing
        assert "anomaly_map" in result
        assert isinstance(result["anomaly_map"], str)  # Should be base64 encoded
