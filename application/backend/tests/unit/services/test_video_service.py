# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from services import VideoService


@pytest.fixture
def fxt_project():
    """Fixture for a test project."""
    return uuid4()


@pytest.fixture
def fxt_video_service():
    """Fixture for VideoService - all methods are static."""
    return VideoService


@pytest.fixture
def fxt_video_bytes():
    """Fixture for test video bytes."""
    return b"fake video content" * 100


@pytest.fixture
def fxt_temp_dir():
    """Fixture for a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestVideoService:
    """Tests for VideoService methods."""

    @pytest.fixture(autouse=True)
    def mock_asyncio_to_thread(self):
        """Mock asyncio.to_thread to run synchronously in tests."""

        async def _mock_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("asyncio.to_thread", side_effect=_mock_to_thread):
            yield

    def test_get_unique_filename_multiple_conflicts(self, fxt_temp_dir):
        """Test filename conflict resolution with multiple existing files."""
        # Create multiple existing files
        Path(os.path.join(fxt_temp_dir, "video.mp4")).touch()
        Path(os.path.join(fxt_temp_dir, "video_1.mp4")).touch()
        Path(os.path.join(fxt_temp_dir, "video_2.mp4")).touch()

        filename = VideoService._get_unique_filename(fxt_temp_dir, "video.mp4")
        assert filename == "video_3.mp4"

    def test_upload_video_success(self, fxt_video_service, fxt_project, fxt_video_bytes):
        """Test successful video upload."""
        mock_bin_repo = MagicMock()
        mock_bin_repo.project_folder_path = "/tmp/test"
        mock_bin_repo.save_file = AsyncMock(return_value="/tmp/test/video.mp4")

        with (
            patch("services.video_service.VideoBinaryRepository", return_value=mock_bin_repo),
            patch("services.video_service.VideoService._get_unique_filename", return_value="video.mp4"),
        ):
            result = asyncio.run(
                fxt_video_service.upload_video(
                    project_id=fxt_project,
                    video_bytes=fxt_video_bytes,
                    original_filename="video.mp4",
                ),
            )

        assert result.filename == "video.mp4"
        assert result.video_path == "/tmp/test/video.mp4"
        assert result.size == len(fxt_video_bytes)
        assert result.project_id == fxt_project
        mock_bin_repo.save_file.assert_called_once_with(filename="video.mp4", content=fxt_video_bytes)

    def test_upload_video_invalid_extension(self, fxt_video_service, fxt_project, fxt_video_bytes):
        """Test upload validation rejects invalid file extensions."""
        with pytest.raises(ValueError, match="Invalid video extension"):
            asyncio.run(
                fxt_video_service.upload_video(
                    project_id=fxt_project,
                    video_bytes=fxt_video_bytes,
                    original_filename="video.txt",
                ),
            )

    def test_upload_video_cleanup_on_error(self, fxt_video_service, fxt_project, fxt_video_bytes):
        """Test cleanup attempts when file save fails."""
        mock_bin_repo = MagicMock()
        mock_bin_repo.project_folder_path = "/tmp/test"
        mock_bin_repo.save_file = AsyncMock(side_effect=OSError("Disk full"))
        mock_bin_repo.delete_file = AsyncMock()

        with (
            patch("services.video_service.VideoBinaryRepository", return_value=mock_bin_repo),
            patch("services.video_service.VideoService._get_unique_filename", return_value="video.mp4"),
            pytest.raises(OSError, match="Disk full"),
        ):
            asyncio.run(
                fxt_video_service.upload_video(
                    project_id=fxt_project,
                    video_bytes=fxt_video_bytes,
                    original_filename="video.mp4",
                ),
            )

        mock_bin_repo.delete_file.assert_called_once_with(filename="video.mp4")

    def test_list_videos_with_files(self, fxt_video_service, fxt_project, fxt_temp_dir):
        """Test listing videos filters non-video files correctly."""
        # Create test video files
        video_files = ["video1.mp4", "video2.avi", "video3.mov"]
        for filename in video_files:
            Path(os.path.join(fxt_temp_dir, filename)).touch()

        # Create a non-video file (should be ignored)
        Path(os.path.join(fxt_temp_dir, "document.txt")).touch()

        mock_bin_repo = MagicMock()
        mock_bin_repo.project_folder_path = fxt_temp_dir

        def get_full_path(filename):
            return os.path.join(fxt_temp_dir, filename)

        mock_bin_repo.get_full_path = get_full_path

        mock_path_instance = MagicMock()
        mock_path_instance.exists = AsyncMock(return_value=True)

        with (
            patch("services.video_service.VideoBinaryRepository", return_value=mock_bin_repo),
            patch("services.video_service.anyio.Path", return_value=mock_path_instance),
        ):
            result = asyncio.run(fxt_video_service.list_videos(project_id=fxt_project))

        assert len(result.videos) == 3
        assert result.pagination.total == 3
        filenames = [v.filename for v in result.videos]
        assert "video1.mp4" in filenames
        assert "video2.avi" in filenames
        assert "video3.mov" in filenames

    def test_get_video_by_filename_found(self, fxt_video_service, fxt_project, fxt_temp_dir):
        """Test getting video by filename when file exists."""
        filename = "test_video.mp4"
        file_path = os.path.join(fxt_temp_dir, filename)
        Path(file_path).touch()

        mock_bin_repo = MagicMock()
        mock_bin_repo.get_full_path = lambda f: os.path.join(fxt_temp_dir, f)

        mock_path_instance = MagicMock()
        mock_path_instance.exists = AsyncMock(return_value=True)

        mock_stats = MagicMock()
        mock_stats.st_size = 12345
        mock_stats.st_ctime = 1625077800.0  # Example timestamp

        with (
            patch("services.video_service.VideoBinaryRepository", return_value=mock_bin_repo),
            patch("services.video_service.anyio.Path", return_value=mock_path_instance),
            patch("services.video_service._validate_filename", return_value=filename),
            patch("services.video_service.os.stat", return_value=mock_stats),
        ):
            result = asyncio.run(fxt_video_service.get_video_by_filename(project_id=fxt_project, filename=filename))

        assert result is not None
        assert result.filename == filename
        assert result.project_id == fxt_project

    def test_get_video_by_filename_not_found(self, fxt_video_service, fxt_project):
        """Test getting video by filename when file doesn't exist."""
        mock_bin_repo = MagicMock()
        mock_bin_repo.get_full_path = lambda f: f"/nonexistent/{f}"

        mock_path_instance = MagicMock()
        mock_path_instance.exists = AsyncMock(return_value=False)

        with (
            patch("services.video_service.VideoBinaryRepository", return_value=mock_bin_repo),
            patch("services.video_service.anyio.Path", return_value=mock_path_instance),
        ):
            result = asyncio.run(
                fxt_video_service.get_video_by_filename(project_id=fxt_project, filename="nonexistent.mp4"),
            )

        assert result is None


# ---------------------------------------------------------------------------
# Regression tests - path traversal (upload path)
# ---------------------------------------------------------------------------


TRAVERSAL_FILENAMES = [
    "../../../../tmp/pwned.mp4",
    "../sibling_project/video.mp4",
    "subdir/../../outside.mp4",
    "/tmp/absolute.mp4",
    "..\\..\\outside.mp4",
    # NUL byte injection
    "video\x00.mp4",
]


class TestUploadVideoPathTraversal:
    """VideoService.upload_video must reject filenames that escape the project dir.

    Regression tests for path traversal issue
    Before the fix, _validate_filename() was called on delete/get paths but
    skipped during upload, allowing arbitrary file writes.
    """

    @pytest.fixture(autouse=True)
    def mock_asyncio_to_thread(self):
        """Run asyncio.to_thread synchronously so the inner closure executes."""

        async def _mock_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("asyncio.to_thread", side_effect=_mock_to_thread):
            yield

    @pytest.mark.parametrize("traversal_filename", TRAVERSAL_FILENAMES)
    def test_upload_rejects_traversal_filename(self, fxt_project, fxt_video_bytes, fxt_temp_dir, traversal_filename):
        """upload_video must raise ValueError for traversal filenames, never call save_file."""
        mock_bin_repo = MagicMock()
        mock_bin_repo.project_folder_path = fxt_temp_dir
        mock_bin_repo.save_file = AsyncMock()

        with (
            patch("services.video_service.VideoBinaryRepository", return_value=mock_bin_repo),
            pytest.raises(ValueError),
        ):
            asyncio.run(
                VideoService.upload_video(
                    project_id=fxt_project,
                    video_bytes=fxt_video_bytes,
                    original_filename=traversal_filename,
                ),
            )

        mock_bin_repo.save_file.assert_not_called()

    def test_upload_bad_payload(self, fxt_project, fxt_video_bytes, tmp_path):
        """Malformed payload must be rejected."""
        project_dir = tmp_path / "data" / "videos" / "projects" / str(fxt_project)
        project_dir.mkdir(parents=True)

        mock_bin_repo = MagicMock()
        mock_bin_repo.project_folder_path = str(project_dir)
        mock_bin_repo.save_file = AsyncMock()

        traversal = "../../anomalib_pwned.mp4"
        with (
            patch("services.video_service.VideoBinaryRepository", return_value=mock_bin_repo),
            pytest.raises(ValueError),
        ):
            asyncio.run(
                VideoService.upload_video(
                    project_id=fxt_project,
                    video_bytes=fxt_video_bytes,
                    original_filename=traversal,
                ),
            )

        assert not (tmp_path / "data" / "videos" / "anomalib_pwned.mp4").exists()
        mock_bin_repo.save_file.assert_not_called()

    def test_upload_accepts_safe_filename(self, fxt_project, fxt_video_bytes, fxt_temp_dir):
        """A clean filename must reach save_file without raising."""
        mock_bin_repo = MagicMock()
        mock_bin_repo.project_folder_path = fxt_temp_dir
        mock_bin_repo.save_file = AsyncMock(return_value=os.path.join(fxt_temp_dir, "video.mp4"))

        with patch("services.video_service.VideoBinaryRepository", return_value=mock_bin_repo):
            result = asyncio.run(
                VideoService.upload_video(
                    project_id=fxt_project,
                    video_bytes=fxt_video_bytes,
                    original_filename="video.mp4",
                ),
            )

        mock_bin_repo.save_file.assert_called_once()
        assert result.filename == "video.mp4"

    def test_delete_video_by_filename_success(self, fxt_video_service, fxt_project):
        """Test successful video deletion."""
        mock_bin_repo = MagicMock()
        mock_bin_repo.get_full_path = lambda f: f"/tmp/{f}"
        mock_bin_repo.delete_file = AsyncMock()

        mock_path_instance = MagicMock()
        mock_path_instance.exists = AsyncMock(return_value=True)

        with (
            patch("services.video_service.VideoBinaryRepository", return_value=mock_bin_repo),
            patch("services.video_service.anyio.Path", return_value=mock_path_instance),
        ):
            asyncio.run(fxt_video_service.delete_video_by_filename(project_id=fxt_project, filename="video.mp4"))

        mock_bin_repo.delete_file.assert_called_once_with(filename="video.mp4")

    def test_delete_video_by_filename_not_found(self, fxt_video_service, fxt_project):
        """Test video deletion raises error when file not found."""
        mock_bin_repo = MagicMock()
        mock_bin_repo.get_full_path = lambda f: f"/nonexistent/{f}"

        mock_path_instance = MagicMock()
        mock_path_instance.exists = AsyncMock(return_value=False)

        with (
            patch("services.video_service.VideoBinaryRepository", return_value=mock_bin_repo),
            patch("services.video_service.anyio.Path", return_value=mock_path_instance),
            pytest.raises(FileNotFoundError),
        ):
            asyncio.run(
                fxt_video_service.delete_video_by_filename(project_id=fxt_project, filename="video.mp4"),
            )

    def test_cleanup_project_videos_success(self, fxt_video_service, fxt_project):
        """Test successful project video cleanup."""
        mock_bin_repo = MagicMock()
        mock_bin_repo.delete_project_folder = AsyncMock()

        with patch("services.video_service.VideoBinaryRepository", return_value=mock_bin_repo):
            asyncio.run(fxt_video_service.cleanup_project_videos(project_id=fxt_project))

        mock_bin_repo.delete_project_folder.assert_called_once()
