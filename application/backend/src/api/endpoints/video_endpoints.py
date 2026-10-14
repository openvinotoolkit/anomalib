# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Endpoints for uploading and managing video files."""

import os
from typing import Annotated
from urllib.parse import unquote
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from api.dependencies import PaginationLimit, get_project_id
from api.endpoints import API_PREFIX
from pydantic_models import Video, VideoList
from services import VideoService

video_api_prefix_url = API_PREFIX + "/projects/{project_id}"
router = APIRouter(
    prefix=video_api_prefix_url,
    tags=["videos"],
)

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_VIDEO_SIZE_BYTES = 10 * 1024**3  # 10 GB


def validate_video_file(file: UploadFile = File(...)) -> UploadFile:
    """Validate the uploaded video file.

    Args:
        file: The uploaded video file.

    Returns:
        The validated upload file.

    Raises:
        HTTPException: If the file is invalid.
    """
    if file.filename is None or "." not in file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Video file must have a filename with extension",
        )

    # Decode percent-encoding (e.g. %00 → \x00, %2F → /) before security checks
    # so that URL-encoded traversal or null-byte payloads are also rejected.
    decoded_filename = unquote(file.filename)
    file.filename = decoded_filename
    # Reject filenames containing path separators, traversal segments, drive qualifiers, or NUL bytes
    safe_name = os.path.basename(decoded_filename)
    drive, _ = os.path.splitdrive(decoded_filename)
    if (
        safe_name != decoded_filename  # contains '/'-separated path components
        or drive  # Windows drive-qualified paths (e.g., C:foo.mp4)
        or "\\" in decoded_filename  # Windows-style separators
        or safe_name in {".", ".."}
        or "\x00" in decoded_filename
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename: path components are not allowed",
        )

    extension = "." + decoded_filename.rsplit(".", maxsplit=1)[-1].lower()
    if extension not in SUPPORTED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid video format. Supported formats: {SUPPORTED_VIDEO_EXTENSIONS}",
        )

    if file.size and file.size > MAX_VIDEO_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Video file too large. Maximum size: {MAX_VIDEO_SIZE_BYTES / (1024**3):.1f} GB",
        )

    return file


@router.post(
    "/videos",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {"description": "Video uploaded successfully", "model": Video},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid video file"},
        status.HTTP_413_CONTENT_TOO_LARGE: {"description": "Video file too large"},
    },
)
async def upload_video(
    project_id: Annotated[UUID, Depends(get_project_id)],
    file: Annotated[UploadFile, Depends(validate_video_file)],
) -> Video:
    """Upload a video file for use as a video source.

    The video is stored on the server and the returned `video_path` can be used
    when creating a video file source.
    """
    video_bytes = await file.read()

    try:
        return await VideoService.upload_video(
            project_id=project_id,
            video_bytes=video_bytes,
            original_filename=file.filename or "video.mp4",
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except OSError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save video: {e}",
        )


@router.get(
    "/videos",
    responses={
        status.HTTP_200_OK: {"description": "List of uploaded videos", "model": VideoList},
    },
)
async def list_videos(
    project_id: Annotated[UUID, Depends(get_project_id)],
    limit: Annotated[int, Depends(PaginationLimit())],
    offset: Annotated[int, Query(ge=0)] = 0,
) -> VideoList:
    """List all uploaded videos for a project."""
    return await VideoService.list_videos(
        project_id=project_id,
        limit=limit,
        offset=offset,
    )


@router.delete(
    "/videos",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Video deleted successfully"},
        status.HTTP_404_NOT_FOUND: {"description": "Video not found"},
    },
)
async def delete_video(
    project_id: Annotated[UUID, Depends(get_project_id)],
    filename: Annotated[str, Query(description="Filename of the video to delete")],
) -> None:
    """Delete an uploaded video by filename."""
    try:
        await VideoService.delete_video_by_filename(project_id=project_id, filename=filename)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
