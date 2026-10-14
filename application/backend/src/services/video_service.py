# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Service for handling video file uploads."""

import os
import re
from datetime import datetime
from uuid import UUID

import anyio
from loguru import logger

from pydantic_models import Video, VideoList
from pydantic_models.base import Pagination
from repositories.binary_repo import VideoBinaryRepository

VALID_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _validate_filename(filename: str, expected_folder: str) -> str:
    """
    Validate that a filename is safe and doesn't allow path traversal.

    Args:
        filename: The filename to validate.
        expected_folder: The folder where the file should reside.

    Returns:
        The validated full path.

    Raises:
        ValueError: If the filename contains path traversal attempts or is invalid.
    """
    # Reject empty filenames
    if not filename:
        raise ValueError("Filename cannot be empty")

    # Reject filenames with NUL bytes
    if "\x00" in filename:
        raise ValueError("Filename cannot contain NUL bytes")

    # Reject filenames with path separators (both POSIX and Windows)
    if os.path.sep in filename or "\\" in filename or (os.path.altsep and os.path.altsep in filename):
        raise ValueError("Filename cannot contain path separators")
    if filename in {".", ".."}:
        raise ValueError("Filename cannot be a bare directory reference")

    # Build the full path and resolve to absolute path
    full_path = os.path.realpath(os.path.join(expected_folder, filename))
    expected_folder_resolved = os.path.realpath(expected_folder)

    # Ensure the resolved path is within the expected folder
    if os.path.commonpath([expected_folder_resolved, full_path]) != expected_folder_resolved:
        raise ValueError("Invalid filename: path traversal detected")

    return full_path


class VideoService:
    """Service for uploading, listing, and deleting video files."""

    @staticmethod
    def _get_unique_filename(folder_path: str, original_filename: str) -> str:
        """
        Get a unique filename, adding number suffix if file already exists.

        Args:
            folder_path: Path to the folder where the file will be saved.
            original_filename: Original filename (e.g., "video.mp4").

        Returns:
            Unique filename (e.g., "video.mp4" or "video_1.mp4" if exists).
        """
        if not os.path.exists(folder_path):
            return original_filename

        name, ext = os.path.splitext(original_filename)
        # Remove any existing numeric suffix to get base name
        base_name = re.sub(r"_\d+$", "", name)

        # Check if file exists
        if not os.path.exists(os.path.join(folder_path, original_filename)):
            return original_filename

        # Find the next available number
        counter = 1
        while True:
            new_filename = f"{base_name}_{counter}{ext}"
            if not os.path.exists(os.path.join(folder_path, new_filename)):
                return new_filename
            counter += 1

    @staticmethod
    async def upload_video(
        project_id: UUID,
        video_bytes: bytes,
        original_filename: str,
    ) -> Video:
        """
        Upload a video file to the server.

        Keeps the original filename. If a file with the same name exists,
        adds a number suffix (e.g., video_1.mp4).

        Args:
            project_id: ID of the project to upload the video to.
            video_bytes: Binary content of the video file.
            original_filename: Original filename to preserve.

        Returns:
            Video metadata including the server-side path.

        Raises:
            ValueError: If the file extension is invalid.
            OSError: If the file cannot be saved.
        """
        # Extract and validate extension
        if not original_filename or "." not in original_filename:
            raise ValueError("Video file must have an extension")

        extension = "." + original_filename.rsplit(".", maxsplit=1)[-1].lower()
        if extension not in VALID_VIDEO_EXTENSIONS:
            raise ValueError(f"Invalid video extension: {extension}. Supported: {VALID_VIDEO_EXTENSIONS}")

        bin_repo = VideoBinaryRepository(project_id=project_id)
        folder_path = bin_repo.project_folder_path

        # Validate the filename to prevent path traversal attacks
        _validate_filename(original_filename, folder_path)

        # Get unique filename (adds suffix if name already taken)
        filename = VideoService._get_unique_filename(folder_path, original_filename)

        try:
            saved_path = await bin_repo.save_file(filename=filename, content=video_bytes)
            logger.info(f"Saved video file: {saved_path}")

            return Video(
                project_id=project_id,
                filename=filename,
                video_path=saved_path,
                size=len(video_bytes),
                created_at=datetime.now(),
            )
        except Exception as e:
            logger.error(f"Failed to save video file: {e}")
            # Attempt cleanup
            try:
                await bin_repo.delete_file(filename=filename)
            except FileNotFoundError as e:
                logger.debug(f"Nothing to cleanup: {e}")
            raise

    @staticmethod
    async def list_videos(project_id: UUID, limit: int = 100, offset: int = 0) -> VideoList:
        """
        List uploaded videos for a project.

        Args:
            project_id: ID of the project.
            limit: Maximum number of videos to return.
            offset: Number of videos to skip.

        Returns:
            Paginated list of videos.
        """
        bin_repo = VideoBinaryRepository(project_id=project_id)
        folder_path = bin_repo.project_folder_path

        videos: list[Video] = []
        total = 0

        if await anyio.Path(folder_path).exists():
            all_files = sorted(os.listdir(folder_path))
            video_files = [f for f in all_files if os.path.splitext(f)[1].lower() in VALID_VIDEO_EXTENSIONS]
            total = len(video_files)

            # Apply pagination
            paginated_files = video_files[offset : offset + limit]

            for filename in paginated_files:
                full_path = bin_repo.get_full_path(filename)
                file_stat = os.stat(full_path)

                videos.append(
                    Video(
                        project_id=project_id,
                        filename=filename,
                        video_path=full_path,
                        size=file_stat.st_size,
                        created_at=datetime.fromtimestamp(file_stat.st_ctime),
                    ),
                )

        return VideoList(
            videos=videos,
            pagination=Pagination(
                limit=limit,
                offset=offset,
                count=len(videos),
                total=total,
            ),
        )

    @staticmethod
    async def get_video_by_filename(project_id: UUID, filename: str) -> Video | None:
        """
        Get a video by its filename.

        Args:
            project_id: ID of the project.
            filename: Name of the video file.

        Returns:
            Video metadata if found, None otherwise.

        Raises:
            ValueError: If the filename contains path traversal attempts.
        """
        bin_repo = VideoBinaryRepository(project_id=project_id)

        # Validate filename to prevent path traversal attacks
        full_path = _validate_filename(filename, bin_repo.project_folder_path)

        if not await anyio.Path(full_path).exists():
            return None

        file_stat = os.stat(full_path)
        return Video(
            project_id=project_id,
            filename=filename,
            video_path=full_path,
            size=file_stat.st_size,
            created_at=datetime.fromtimestamp(file_stat.st_ctime),
        )

    @staticmethod
    async def delete_video_by_filename(project_id: UUID, filename: str) -> None:
        """
        Delete an uploaded video by filename.

        Args:
            project_id: ID of the project.
            filename: Name of the video file to delete.

        Raises:
            FileNotFoundError: If the video file is not found.
            ValueError: If the filename contains path traversal attempts.
        """
        bin_repo = VideoBinaryRepository(project_id=project_id)

        # Validate filename to prevent path traversal attacks
        full_path = _validate_filename(filename, bin_repo.project_folder_path)

        if not await anyio.Path(full_path).exists():
            raise FileNotFoundError(f"Video '{filename}' not found")

        await bin_repo.delete_file(filename=filename)
        logger.info(f"Deleted video file: {filename}")

    @staticmethod
    async def cleanup_project_videos(project_id: UUID) -> None:
        """
        Delete all videos for a project.

        Args:
            project_id: ID of the project.
        """
        try:
            bin_repo = VideoBinaryRepository(project_id=project_id)
            await bin_repo.delete_project_folder()
            logger.info(f"Cleaned up video files for project {project_id}")
        except Exception as e:
            logger.warning(f"Failed to cleanup video files for project {project_id}: {e}")
