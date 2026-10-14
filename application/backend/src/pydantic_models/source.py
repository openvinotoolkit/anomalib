# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from os import getenv
from typing import Annotated, Literal
from urllib.parse import urlparse, urlunparse
from uuid import UUID

from pydantic import BaseModel, Field, TypeAdapter

from pydantic_models.base import BaseIDNameModel, NoRequiredIDs, Pagination

IP_CAMERA_USERNAME = "IP_CAMERA_USERNAME"
IP_CAMERA_PASSWORD = "IP_CAMERA_PASSWORD"  # noqa: S105  # nosec B105


class SourceType(StrEnum):
    # TODO: remove or refactor "DISCONNECTED" into separate enums if needed
    DISCONNECTED = "disconnected"
    USB_CAMERA = "usb_camera"
    IP_CAMERA = "ip_camera"
    VIDEO_FILE = "video_file"
    IMAGES_FOLDER = "images_folder"


class BaseSourceConfig(BaseIDNameModel):
    project_id: UUID
    source_type: str


class DisconnectedSourceConfig(BaseSourceConfig):
    source_type: Literal[SourceType.DISCONNECTED] = SourceType.DISCONNECTED
    project_id: UUID = UUID("00000000-0000-0000-0000-000000000000")
    name: str = "No Source"


class UsbCameraSourceConfig(BaseSourceConfig):
    source_type: Literal[SourceType.USB_CAMERA]
    device_id: int

    model_config = {
        "json_schema_extra": {
            "example": {
                "source_type": "usb_camera",
                "name": "USB Camera 0",
                "id": "1fe8017e-3494-407d-9c80-d3def3ab73be",
                "device_id": 0,
            },
        },
    }


class IPCameraSourceConfig(BaseSourceConfig):
    source_type: Literal[SourceType.IP_CAMERA]
    stream_url: str
    auth_required: bool = False

    model_config = {
        "json_schema_extra": {
            "example": {
                "source_type": "ip_camera",
                "name": "Street Camera 123",
                "id": "b7e0a318-e269-46fc-82b8-6b9a485d9162",
                "stream_url": "http://example.com/stream",
                "auth_required": True,
            },
        },
    }

    def get_configured_stream_url(self) -> str:
        """Configure stream URL with authentication if required."""
        if not self.auth_required:
            return self.stream_url

        username = getenv(IP_CAMERA_USERNAME)
        password = getenv(IP_CAMERA_PASSWORD)

        if not username or not password:
            raise RuntimeError("IP camera credentials not provided.")

        # Modify the stream URL to include authentication
        uri = urlparse(self.stream_url)
        netloc = f"{username}:{password}@{uri.netloc}"
        return urlunparse((uri.scheme, netloc, uri.path, uri.params, uri.query, uri.fragment))


class VideoFileSourceConfig(BaseSourceConfig):
    source_type: Literal[SourceType.VIDEO_FILE]
    video_path: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "source_type": "video_file",
                "name": "Sample Video",
                "id": "d1a28f9f-8458-4615-bda0-5c5ebc649e9a",
                "video_path": "/path/to/video.mp4",
            },
        },
    }


class ImagesFolderSourceConfig(BaseSourceConfig):
    source_type: Literal[SourceType.IMAGES_FOLDER]
    images_folder_path: str
    ignore_existing_images: bool

    model_config = {
        "json_schema_extra": {
            "example": {
                "source_type": "images_folder",
                "name": "Best Photos",
                "id": "926063c5-f9ea-4a68-b52f-46e62d050d76",
                "images_folder_path": "/path/to/images",
                "ignore_existing_images": True,
            },
        },
    }


Source = Annotated[
    UsbCameraSourceConfig
    | IPCameraSourceConfig
    | VideoFileSourceConfig
    | ImagesFolderSourceConfig
    | DisconnectedSourceConfig,
    Field(discriminator="source_type"),
]

SourceAdapter: TypeAdapter[Source] = TypeAdapter(Source)


class SourceList(BaseModel):
    sources: list[Source]
    pagination: Pagination


# ---------------------------------
# Creation Schemas (POST requests)
# ---------------------------------
# These schemas inherit from HasID first to override the required ID field with an auto-generated one (if absent) via
# MRO (Method Resolution Order).


class UsbCameraSourceConfigCreate(NoRequiredIDs, UsbCameraSourceConfig):
    pass


class IPCameraSourceConfigCreate(NoRequiredIDs, IPCameraSourceConfig):
    pass


class VideoFileSourceConfigCreate(NoRequiredIDs, VideoFileSourceConfig):
    pass


class ImagesFolderSourceConfigCreate(NoRequiredIDs, ImagesFolderSourceConfig):
    pass


SourceCreate = Annotated[
    UsbCameraSourceConfigCreate
    | IPCameraSourceConfigCreate
    | VideoFileSourceConfigCreate
    | ImagesFolderSourceConfigCreate,
    Field(discriminator="source_type"),
]

SourceCreateAdapter: TypeAdapter[SourceCreate] = TypeAdapter(SourceCreate)
