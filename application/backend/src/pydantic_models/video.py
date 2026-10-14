# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field

from pydantic_models.base import Pagination


class VideoExtension(StrEnum):
    MP4 = ".mp4"
    AVI = ".avi"
    MOV = ".mov"
    MKV = ".mkv"
    WEBM = ".webm"


class Video(BaseModel):
    """Model representing an uploaded video file.

    Videos are identified by their filename (unique within a project folder).
    No database persistence - metadata is derived from the filesystem.
    """

    project_id: UUID
    filename: str = Field(..., description="Filename (unique identifier within project)")
    video_path: str = Field(..., description="Full path to the video file on the server")
    size: int = Field(..., ge=0, description="File size in bytes")
    created_at: datetime | None = Field(None, description="File creation timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "project_id": "16e07d18-196e-4e33-bf98-ac1d35dcaaaa",
                "filename": "sample_video.mp4",
                "video_path": "data/videos/projects/16e07d18-196e-4e33-bf98-ac1d35dcaaaa/sample_video.mp4",
                "size": 10485760,
                "created_at": "2025-01-21T10:30:00",
            },
        },
    }


class VideoList(BaseModel):
    """Paginated list of videos."""

    videos: list[Video]
    pagination: Pagination
