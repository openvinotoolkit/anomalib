# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pandas as pd
import pytest

from pydantic_models import DatasetSnapshot, Media
from services.dataset_snapshot_service import DatasetSnapshotService


@pytest.fixture
def fxt_project_id():
    return uuid4()


@pytest.fixture
def fxt_media(fxt_project_id):
    return Media(
        id=uuid4(),
        project_id=fxt_project_id,
        filename="test.jpg",
        size=100,
        is_anomalous=False,
        width=100,
        height=100,
    )


@pytest.fixture
def fxt_snapshot(fxt_project_id):
    return DatasetSnapshot(
        id=uuid4(),
        project_id=fxt_project_id,
        filename="snapshot.parquet",
        count=1,  # Add missing required field
    )


def test_create_snapshot_success(fxt_project_id, fxt_media, fxt_snapshot):
    """Test creating a snapshot successfully."""

    async def _test():
        with (
            patch("services.dataset_snapshot_service.get_async_db_session_ctx") as mock_db_ctx,
            patch("services.dataset_snapshot_service.MediaRepository") as mock_media_repo_cls,
            patch("services.dataset_snapshot_service.ImageBinaryRepository") as mock_image_repo_cls,
            patch("services.dataset_snapshot_service.DatasetSnapshotBinaryRepository") as mock_snapshot_bin_repo_cls,
            patch("services.dataset_snapshot_service.DatasetSnapshotRepository") as mock_snapshot_repo_cls,
            patch("services.dataset_snapshot_service.asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
        ):
            # Mock DB session
            mock_session = AsyncMock()
            mock_db_ctx.return_value.__aenter__.return_value = mock_session

            # Mock Media Repo
            mock_media_repo = MagicMock()

            async def mock_get_all_streaming():
                yield fxt_media

            mock_media_repo.get_all_streaming = mock_get_all_streaming
            mock_media_repo_cls.return_value = mock_media_repo

            # Mock Image Binary Repo
            mock_image_repo = MagicMock()
            mock_image_repo.read_file = AsyncMock(return_value=b"fake_image_bytes")
            mock_image_repo_cls.return_value = mock_image_repo

            # Mock Parquet writing
            mock_to_thread.return_value = b"parquet_content"

            # Mock Snapshot Binary Repo
            mock_snapshot_bin_repo = MagicMock()
            mock_snapshot_bin_repo.save_file = AsyncMock()
            mock_snapshot_bin_repo_cls.return_value = mock_snapshot_bin_repo

            # Mock Snapshot Repo
            mock_snapshot_repo = MagicMock()
            mock_snapshot_repo.save = AsyncMock(return_value=fxt_snapshot)
            mock_snapshot_repo_cls.return_value = mock_snapshot_repo

            result = await DatasetSnapshotService.create_snapshot(fxt_project_id)

            assert result == fxt_snapshot
            mock_image_repo.read_file.assert_called_once_with(fxt_media.filename)
            mock_snapshot_bin_repo.save_file.assert_called_once()
            mock_snapshot_repo.save.assert_called_once()

    asyncio.run(_test())


def test_use_snapshot_as_folder(fxt_project_id):
    """Test context manager for using snapshot as folder."""

    async def _test():
        snapshot_id = uuid4()

        with (
            patch("services.dataset_snapshot_service.DatasetSnapshotBinaryRepository") as mock_bin_repo_cls,
            patch("services.dataset_snapshot_service.asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread,
            patch("tempfile.TemporaryDirectory") as mock_temp_dir_cls,
        ):
            mock_bin_repo = MagicMock()
            mock_bin_repo.get_snapshot_path.return_value = "/path/to/snapshot.parquet"
            mock_bin_repo_cls.return_value = mock_bin_repo

            mock_temp_dir = MagicMock()
            mock_temp_dir.__enter__.return_value = "/tmp/dir"
            mock_temp_dir_cls.return_value = mock_temp_dir

            async with DatasetSnapshotService.use_snapshot_as_folder(snapshot_id, fxt_project_id) as path:
                assert path == "/tmp/dir"
                mock_to_thread.assert_called_once()
                # Verify extract_snapshot_to_path was called via to_thread
                call_args = mock_to_thread.call_args
                assert call_args[0][0] == DatasetSnapshotService.extract_snapshot_to_path
                assert call_args[0][1] == "/path/to/snapshot.parquet"
                assert call_args[0][2] == "/tmp/dir"

    asyncio.run(_test())


def test_extract_snapshot_to_path():
    """Test extracting snapshot to path."""
    snapshot_path = "dummy.parquet"
    temp_dir = "/tmp/dataset"

    # Create a dummy dataframe batch
    df = pd.DataFrame({"image": [b"fake_bytes"], "is_anomalous": [False], "filename": ["img.jpg"]})

    # Mock row tuple to behave like namedtuple
    # In iter_tuples, rows are namedtuples by default

    # We need to mock pyarrow.parquet.ParquetFile
    with (
        patch("pyarrow.parquet.ParquetFile") as mock_parquet_file_cls,
        patch("os.makedirs") as mock_makedirs,
        patch("PIL.Image.open") as mock_image_open,
        patch("io.BytesIO"),
        patch("os.path.exists", return_value=False),
    ):
        # Setup ParquetFile mock
        mock_parquet_file = MagicMock()
        mock_parquet_file_cls.return_value = mock_parquet_file

        # Setup iter_batches to return one batch
        mock_batch = MagicMock()
        mock_batch.to_pandas.return_value = df
        mock_parquet_file.iter_batches.return_value = [mock_batch]

        mock_img = MagicMock()
        mock_image_open.return_value = mock_img

        DatasetSnapshotService.extract_snapshot_to_path(snapshot_path, temp_dir)

        mock_parquet_file_cls.assert_called_once_with(snapshot_path)
        mock_parquet_file.iter_batches.assert_called_once()
        mock_makedirs.assert_called()
        mock_image_open.assert_called_once()
        mock_img.save.assert_called_once()
