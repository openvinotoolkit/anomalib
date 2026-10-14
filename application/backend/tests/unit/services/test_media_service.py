# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image

from repositories import MediaRepository
from repositories.binary_repo import ImageBinaryRepository
from services import MediaService, ResourceNotFoundError
from services.exceptions import ResourceType


@pytest.fixture
def fxt_media_repository():
    """Fixture for a mock media repository."""
    return MagicMock(spec=MediaRepository)


@pytest.fixture
def fxt_image_binary_repo():
    """Fixture for a mock image binary repository."""
    return MagicMock(spec=ImageBinaryRepository)


@pytest.fixture
def fxt_media_service():
    """Fixture for MediaService - all methods are static."""
    return MediaService


@pytest.fixture(autouse=True)
def mock_db_context():
    """Mock the database context for all tests."""
    with patch("services.media_service.get_async_db_session_ctx") as mock_db_ctx:
        mock_session = AsyncMock()
        mock_db_ctx.return_value.__aenter__.return_value = mock_session
        mock_db_ctx.return_value.__aexit__.return_value = None
        yield mock_db_ctx


class TestMediaService:
    @pytest.fixture(autouse=True)
    def mock_asyncio_to_thread(self):
        """Mock asyncio.to_thread to run synchronously in tests."""

        async def _mock_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("asyncio.to_thread", side_effect=_mock_to_thread):
            yield

    def test_get_media_list(self, fxt_media_service, fxt_media_repository, fxt_media_list, fxt_project):
        """Test getting media list."""
        fxt_media_repository.get_all_count = AsyncMock(return_value=len(fxt_media_list.media))
        fxt_media_repository.get_all_pagination = AsyncMock(return_value=fxt_media_list.media)

        with patch("services.media_service.MediaRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_media_repository

            result = asyncio.run(fxt_media_service.get_media_list(fxt_project.id, limit=20, offset=0))

        assert result.media == fxt_media_list.media
        assert result.pagination.total == len(fxt_media_list.media)
        fxt_media_repository.get_all_count.assert_called_once()
        fxt_media_repository.get_all_pagination.assert_called_once_with(limit=20, offset=0)

    def test_get_media_by_id(self, fxt_media_service, fxt_media_repository, fxt_media, fxt_project):
        """Test getting media by ID."""
        fxt_media_repository.get_by_id.return_value = fxt_media

        with patch("services.media_service.MediaRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_media_repository

            result = asyncio.run(fxt_media_service.get_media_by_id(fxt_project.id, fxt_media.id))

        assert result == fxt_media
        fxt_media_repository.get_by_id.assert_called_once_with(fxt_media.id)

    def test_get_media_by_id_not_found(self, fxt_media_service, fxt_media_repository, fxt_project):
        """Test getting media by ID when not found."""
        fxt_media_repository.get_by_id.return_value = None

        with patch("services.media_service.MediaRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_media_repository

            result = asyncio.run(fxt_media_service.get_media_by_id(fxt_project.id, "non-existent-id"))

        assert result is None
        fxt_media_repository.get_by_id.assert_called_once_with("non-existent-id")

    def test_get_media_file_path_success(self, fxt_media_service, fxt_media, fxt_project):
        """Test getting media file path successfully."""
        with patch("services.media_service.ImageBinaryRepository") as mock_bin_repo_class:
            mock_bin_repo = MagicMock()
            mock_bin_repo_class.return_value = mock_bin_repo
            mock_bin_repo.get_full_path.return_value = "/path/to/file.jpg"

            # Mock the get_media_by_id method
            fxt_media_service.get_media_by_id = AsyncMock(return_value=fxt_media)

            result = asyncio.run(fxt_media_service.get_media_file_path(fxt_project.id, fxt_media.id))

            assert result == "/path/to/file.jpg"
            mock_bin_repo.get_full_path.assert_called_once_with(filename=fxt_media.filename)

    def test_get_media_file_path_not_found(self, fxt_media_service, fxt_project):
        """Test getting media file path when media not found."""
        fxt_media_service.get_media_by_id = AsyncMock(return_value=None)

        with pytest.raises(FileNotFoundError) as exc_info:
            asyncio.run(fxt_media_service.get_media_file_path(fxt_project.id, "non-existent-id"))

        assert "Media with ID non-existent-id not found" in str(exc_info.value)

    def test_upload_image_success(self, fxt_media_service, fxt_upload_file, fxt_project, fxt_image_bytes):
        """Test successful image upload (no inline thumbnail generation)."""
        mock_bin_repo = MagicMock()
        mock_bin_repo.save_file = AsyncMock(return_value="/path/to/file.jpg")

        mock_media_repo = MagicMock()
        mock_media_repo.save = AsyncMock(return_value=MagicMock())

        with (
            patch("services.media_service.ImageBinaryRepository", return_value=mock_bin_repo),
            patch("services.media_service.MediaRepository", return_value=mock_media_repo),
        ):
            result = asyncio.run(
                fxt_media_service.upload_image(fxt_project.id, fxt_image_bytes, False, extension=".jpg"),
            )

        assert result is not None
        assert mock_bin_repo.save_file.call_count == 1  # only original saved
        mock_media_repo.save.assert_called_once()

    def test_upload_image_numpy_success(self, fxt_media_service, fxt_project):
        """Test successful image upload with numpy array."""
        # Create a random numpy image (100x100 RGB)
        numpy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        mock_bin_repo = MagicMock()
        mock_bin_repo.save_file = AsyncMock(return_value="/path/to/file.png")

        mock_media_repo = MagicMock()
        mock_media_repo.save = AsyncMock(return_value=MagicMock())

        with (
            patch("services.media_service.ImageBinaryRepository", return_value=mock_bin_repo),
            patch("services.media_service.MediaRepository", return_value=mock_media_repo),
        ):
            result = asyncio.run(fxt_media_service.upload_image(fxt_project.id, numpy_image, False, extension=".png"))

        assert result is not None
        assert mock_bin_repo.save_file.call_count == 1
        mock_media_repo.save.assert_called_once()

    def test_upload_image_rollback_on_error(self, fxt_media_service, fxt_upload_file, fxt_project, fxt_image_bytes):
        """Test image upload rollback on error."""
        mock_bin_repo = MagicMock()
        mock_bin_repo.save_file = AsyncMock(return_value="/path/to/file.jpg")
        mock_bin_repo.delete_file = AsyncMock()

        mock_media_repo = MagicMock()
        mock_media_repo.save = AsyncMock(side_effect=Exception("Database error"))

        with (
            patch("services.media_service.ImageBinaryRepository", return_value=mock_bin_repo),
            patch("services.media_service.MediaRepository", return_value=mock_media_repo),
            pytest.raises(Exception, match="Database error"),
        ):
            asyncio.run(fxt_media_service.upload_image(fxt_project.id, fxt_image_bytes, False, extension=".jpg"))

        # Verify rollback deletes both files
        assert mock_bin_repo.delete_file.call_count == 1

    def test_upload_image_rollback_file_not_found(
        self,
        fxt_media_service,
        fxt_upload_file,
        fxt_project,
        fxt_image_bytes,
    ):
        """Test image upload rollback when file deletion fails with FileNotFoundError."""
        mock_bin_repo = MagicMock()
        mock_bin_repo.save_file = AsyncMock(return_value="/path/to/file.jpg")
        mock_bin_repo.delete_file = AsyncMock(side_effect=FileNotFoundError())

        mock_media_repo = MagicMock()
        mock_media_repo.save = AsyncMock(side_effect=Exception("Database error"))

        with (
            patch("services.media_service.ImageBinaryRepository", return_value=mock_bin_repo),
            patch("services.media_service.MediaRepository", return_value=mock_media_repo),
            pytest.raises(Exception, match="Database error"),
        ):
            asyncio.run(fxt_media_service.upload_image(fxt_project.id, fxt_image_bytes, False, extension=".jpg"))

        # Verify rollback attempted to delete both files
        assert mock_bin_repo.delete_file.call_count == 1

    def test_delete_media_success(self, fxt_media_service, fxt_media, fxt_project):
        """Test successful media deletion."""
        mock_bin_repo = MagicMock()
        mock_bin_repo.delete_file = AsyncMock()

        mock_media_repo = MagicMock()
        mock_media_repo.get_by_id = AsyncMock(return_value=fxt_media)
        mock_media_repo.delete_by_id = AsyncMock()

        with (
            patch("services.media_service.ImageBinaryRepository", return_value=mock_bin_repo),
            patch("services.media_service.MediaRepository", return_value=mock_media_repo),
        ):
            asyncio.run(fxt_media_service.delete_media(fxt_media.id, fxt_project.id))

        # Should delete both original and thumbnail
        assert mock_bin_repo.delete_file.call_count == 2
        calls = mock_bin_repo.delete_file.call_args_list
        assert calls[0][1]["filename"] == fxt_media.filename
        assert calls[1][1]["filename"] == f"thumb_{fxt_media.id}.png"
        mock_media_repo.delete_by_id.assert_called_once_with(fxt_media.id)

    def test_delete_media_not_found(self, fxt_media_service, fxt_project):
        """Test media deletion when media not found."""
        # Mock the repository method
        mock_media_repo = MagicMock()
        mock_media_repo.get_by_id = AsyncMock(return_value=None)

        with patch("services.media_service.MediaRepository") as mock_repo_class:
            mock_repo_class.return_value = mock_media_repo

            with pytest.raises(ResourceNotFoundError) as exc_info:
                asyncio.run(fxt_media_service.delete_media(uuid4(), fxt_project.id))

            assert exc_info.value.resource_type == ResourceType.MEDIA

        mock_media_repo.get_by_id.assert_called_once()

    def test_delete_media_file_not_found(self, fxt_media_service, fxt_media, fxt_project):
        """Test media deletion when file not found."""
        mock_bin_repo = MagicMock()
        mock_bin_repo.delete_file = AsyncMock(side_effect=FileNotFoundError())

        mock_media_repo = MagicMock()
        mock_media_repo.get_by_id = AsyncMock(return_value=fxt_media)
        mock_media_repo.delete_by_id = AsyncMock()

        with (
            patch("services.media_service.ImageBinaryRepository", return_value=mock_bin_repo),
            patch("services.media_service.MediaRepository", return_value=mock_media_repo),
        ):
            # Should not raise an exception, just log warning
            asyncio.run(fxt_media_service.delete_media(fxt_media.id, fxt_project.id))

        # Should try to delete both files (both raise FileNotFoundError)
        assert mock_bin_repo.delete_file.call_count == 2

    def test_delete_media_other_error(self, fxt_media_service, fxt_media, fxt_project):
        """Test media deletion with other error."""
        with patch("services.media_service.ImageBinaryRepository") as mock_bin_repo_class:
            mock_bin_repo = MagicMock()
            mock_bin_repo_class.return_value = mock_bin_repo
            mock_bin_repo.delete_file = AsyncMock(side_effect=Exception("File system error"))

            # Mock the repository method
            mock_media_repo = MagicMock()
            mock_media_repo.get_by_id = AsyncMock(return_value=fxt_media)
            mock_media_repo.delete_by_id = AsyncMock()

            with patch("services.media_service.MediaRepository") as mock_repo_class:
                mock_repo_class.return_value = mock_media_repo

                with pytest.raises(Exception, match="File system error"):
                    asyncio.run(fxt_media_service.delete_media(fxt_media.id, fxt_project.id))

    def test_thumbnail_generation_success(self, fxt_media_service, fxt_project):
        """Test thumbnail generation saves PNG and uses expected filename."""
        # Create valid image bytes
        img = Image.new("RGB", (512, 512), color="red")
        with BytesIO() as buf:
            img.save(buf, format="JPEG")
            image_bytes = buf.getvalue()

        mock_bin_repo = MagicMock()

        # Capture arguments to validate content and filename
        async def _save_file(filename: str, content: bytes) -> str:  # type: ignore[override]
            assert filename.startswith("thumb_") and filename.endswith(".png")
            assert content.startswith(b"\x89PNG")
            return "/path/to/thumb.png"

        mock_bin_repo.save_file = AsyncMock(side_effect=_save_file)

        media_id = uuid4()

        with patch("services.media_service.ImageBinaryRepository", return_value=mock_bin_repo):
            asyncio.run(
                fxt_media_service.generate_thumbnail(
                    project_id=fxt_project.id,
                    media_id=media_id,
                    image=image_bytes,
                ),
            )

    def test_thumbnail_upload_and_deletion(self, fxt_media_service, fxt_upload_file, fxt_project):
        """Test manual thumbnail generation after upload and deletion of both files."""
        mock_bin_repo = MagicMock()
        mock_bin_repo.save_file = AsyncMock(return_value="/path/to/file.jpg")
        mock_bin_repo.delete_file = AsyncMock()

        mock_media_repo = MagicMock()
        saved_media = MagicMock(id=uuid4(), filename="test.jpg")
        mock_media_repo.save = AsyncMock(return_value=saved_media)
        mock_media_repo.get_by_id = AsyncMock(return_value=saved_media)
        mock_media_repo.delete_by_id = AsyncMock()

        # Valid image bytes for thumbnail generation
        img = Image.new("RGB", (512, 512), color="blue")
        with BytesIO() as buf:
            img.save(buf, format="JPEG")
            image_bytes = buf.getvalue()

        with (
            patch("services.media_service.ImageBinaryRepository", return_value=mock_bin_repo),
            patch("services.media_service.MediaRepository", return_value=mock_media_repo),
        ):
            # Upload saves only original
            result = asyncio.run(fxt_media_service.upload_image(fxt_project.id, image_bytes, False, extension=".jpg"))
            assert result is not None
            assert mock_bin_repo.save_file.call_count == 1

            # Generate and save thumbnail (background-equivalent)
            asyncio.run(
                fxt_media_service.generate_thumbnail(
                    project_id=fxt_project.id,
                    media_id=saved_media.id,
                    image=image_bytes,
                ),
            )

            # Deletion removes both files
            asyncio.run(fxt_media_service.delete_media(saved_media.id, fxt_project.id))
            assert mock_bin_repo.delete_file.call_count == 2
