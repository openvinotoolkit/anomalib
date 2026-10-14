# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from anomalib.deploy import CompressionType, ExportType

from pydantic_models import Model
from pydantic_models.model import ExportParameters
from services import ResourceNotFoundError
from services.model_service import ModelService


@pytest.fixture
def mock_model():
    id = uuid4()
    return Model(
        id=id,
        project_id=uuid4(),
        name=f"Padim ({str(id).split('-')[0]})",
        architecture="padim",
        format=ExportType.OPENVINO,
        is_ready=True,
        train_job_id=uuid4(),
        dataset_snapshot_id=uuid4(),
    )


@pytest.fixture
def mock_model_service():
    return ModelService()


@pytest.mark.asyncio
@patch("services.model_service.ModelService.get_model_by_id")
@patch("services.model_service.ModelExportBinaryRepository")
@patch("services.model_service.anyio.Path")
@patch("services.model_service.ModelBinaryRepository")
@patch("services.model_service.asyncio.to_thread")
async def test_export_model_success(
    mock_to_thread,
    mock_model_binary_repo_cls,
    mock_anyio_path_cls,
    mock_model_export_repo_cls,
    mock_get_model_by_id,
    mock_model_service,
    mock_model,
    tmp_path,
):
    """Test successful model export."""
    # Setup mocks
    mock_get_model_by_id.return_value = mock_model

    # Setup binary repo mock
    mock_binary_repo = MagicMock()
    mock_binary_repo.model_folder_path = str(tmp_path)
    mock_model_binary_repo_cls.return_value = mock_binary_repo

    # Setup export repo mock
    mock_export_repo = MagicMock()
    export_zip_str = str(tmp_path / "exports" / "Padim.zip")
    mock_export_repo.get_model_export_path.return_value = export_zip_str
    mock_model_export_repo_cls.return_value = mock_export_repo

    # anyio.Path mock for export zip path handling
    mock_anyio_path = MagicMock()
    mock_anyio_path.exists = AsyncMock(return_value=False)
    mock_anyio_path_cls.return_value = mock_anyio_path

    # Create dummy checkpoint file
    ckpt_dir = tmp_path / "checkpoint"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "model.ckpt").touch()

    # Setup export params
    export_params = ExportParameters(format=ExportType.OPENVINO, compression=CompressionType.FP16)

    # Setup to_thread result to match expected return
    mock_to_thread.return_value = mock_anyio_path

    # Call method
    result = await mock_model_service.export_model(mock_model.project_id, mock_model.id, export_params)

    # Verify results
    assert result == mock_anyio_path
    mock_model_export_repo_cls.assert_called_once_with(project_id=mock_model.project_id, model_id=mock_model.id)
    mock_anyio_path_cls.assert_called_once_with(export_zip_str)
    mock_to_thread.assert_called_once()


@pytest.mark.asyncio
@patch("services.model_service.ModelService.get_model_by_id")
async def test_export_model_not_found(mock_get_model_by_id, mock_model_service):
    """Test export when model does not exist."""
    mock_get_model_by_id.return_value = None

    export_params = ExportParameters(format=ExportType.OPENVINO)

    with pytest.raises(ResourceNotFoundError):
        await mock_model_service.export_model(uuid4(), uuid4(), export_params)


@pytest.mark.asyncio
@patch("services.model_service.ModelService.get_model_by_id")
@patch("services.model_service.ModelExportBinaryRepository")
@patch("services.model_service.anyio.Path")
@patch("services.model_service.ModelBinaryRepository")
async def test_export_model_ckpt_not_found(
    mock_model_binary_repo_cls,
    mock_anyio_path_cls,
    mock_model_export_repo_cls,
    mock_get_model_by_id,
    mock_model_service,
    mock_model,
    tmp_path,
):
    """Test export when checkpoint file is missing."""
    mock_get_model_by_id.return_value = mock_model

    mock_binary_repo = MagicMock()
    mock_binary_repo.model_folder_path = str(tmp_path)
    mock_model_binary_repo_cls.return_value = mock_binary_repo

    mock_export_repo = MagicMock()
    mock_export_repo.get_model_export_path.return_value = str(tmp_path / "exports" / "Padim.zip")
    mock_model_export_repo_cls.return_value = mock_export_repo

    mock_export_path = MagicMock()
    mock_export_path.exists = AsyncMock(return_value=False)
    mock_anyio_path_cls.return_value = mock_export_path

    # Checkpoint file intentionally omitted to test error handling

    export_params = ExportParameters(format=ExportType.OPENVINO)

    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        await mock_model_service.export_model(mock_model.project_id, mock_model.id, export_params)


@pytest.mark.asyncio
@patch("services.model_service.ModelService.get_model_by_id")
@patch("services.model_service.ModelExportBinaryRepository")
@patch("services.model_service.anyio.Path")
@patch("services.model_service.asyncio.to_thread")
async def test_export_model_cached_logic(
    mock_to_thread,
    mock_anyio_path_cls,
    mock_model_export_repo_cls,
    mock_get_model_by_id,
    mock_model_service,
    mock_model,
):
    """Test that cached export is returned if it exists."""
    mock_get_model_by_id.return_value = mock_model

    export_params = ExportParameters(format=ExportType.OPENVINO)

    mock_export_repo = MagicMock()
    mock_export_repo.get_model_export_path.return_value = "/tmp/export.zip"
    mock_model_export_repo_cls.return_value = mock_export_repo

    mock_export_path = MagicMock()
    mock_export_path.exists = AsyncMock(return_value=True)
    mock_anyio_path_cls.return_value = mock_export_path

    await mock_model_service.export_model(mock_model.project_id, mock_model.id, export_params)

    # Should return the cached path immediately without triggering export
    mock_to_thread.assert_not_called()
