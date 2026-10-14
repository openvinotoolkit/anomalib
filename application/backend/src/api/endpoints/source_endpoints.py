# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Endpoints for managing pipeline sources"""

from typing import Annotated
from uuid import UUID

import yaml
from fastapi import APIRouter, Body, Depends, File, Query, UploadFile, status
from fastapi.exceptions import HTTPException
from fastapi.openapi.models import Example
from fastapi.responses import FileResponse, Response

from api.dependencies import PaginationLimit, get_configuration_service, get_project_id, get_source_id
from pydantic_models import Source, SourceType
from pydantic_models.source import SourceAdapter, SourceCreate, SourceCreateAdapter, SourceList
from services import ConfigurationService, ResourceAlreadyExistsError, ResourceInUseError, ResourceNotFoundError

router = APIRouter(prefix="/api/projects/{project_id}/sources", tags=["Sources"])

CREATE_SOURCE_BODY_DESCRIPTION = """
Configuration for the new source. The exact list of fields that can be configured depends on the source type.
"""
CREATE_SOURCE_BODY_EXAMPLES = {
    "webcam": Example(
        summary="Webcam source",
        description="Configuration for a webcam source",
        value={
            "source_type": "webcam",
            "name": "My Webcam",
            "device_id": 0,
        },
    ),
    "ip_camera": Example(
        summary="IP camera source",
        description="Configuration for an IP camera source",
        value={
            "source_type": "ip_camera",
            "name": "IP Camera 1",
            "stream_url": "rtsp://192.168.1.100:554/stream1",
            "auth_required": True,
        },
    ),
    "video_file": Example(
        summary="Video file source",
        description="Configuration for a video file source",
        value={
            "source_type": "video_file",
            "name": "Camera recording 123",
            "video_path": "/path/to/video.mp4",
        },
    ),
    "images_folder": Example(
        summary="Images folder source",
        description="Configuration for a folder containing images source",
        value={
            "source_type": "images_folder",
            "name": "Production Samples",
            "folder_path": "/path/to/images",
            "ignore_existing_images": True,
        },
    ),
}

UPDATE_SOURCE_BODY_DESCRIPTION = """
Partial source configuration update. May contain any subset of fields from the respective source type
(e.g., 'device_id' for webcams; 'video_path' for video files).
Fields not included in the request will remain unchanged. The 'source_type' field cannot be changed.
"""
UPDATE_SOURCE_BODY_EXAMPLES = {
    "webcam": Example(
        summary="Update webcam source",
        description="Rename a webcam source",
        value={
            "name": "Updated Webcam Name",
        },
    ),
    "video_file": Example(
        summary="Update video file source",
        description="Change the video path for a video file source",
        value={
            "video_path": "/new/path/to/video.mp4",
        },
    ),
}


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {"description": "Source created", "model": Source},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid source ID or request body"},
        status.HTTP_409_CONFLICT: {"description": "Source already exists"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Source could not be connected to"},
    },
)
async def create_source(
    project_id: Annotated[UUID, Depends(get_project_id)],
    source_config: Annotated[
        SourceCreate,
        Body(description=CREATE_SOURCE_BODY_DESCRIPTION, openapi_examples=CREATE_SOURCE_BODY_EXAMPLES),
    ],
    configuration_service: Annotated[ConfigurationService, Depends(get_configuration_service)],
) -> Source:
    """Create and configure a new source"""
    # Inject project_id from URL path into the config
    source_config.project_id = project_id

    # Validate the complete config
    validated_source = SourceCreateAdapter.validate_python(source_config)

    if validated_source.source_type == SourceType.DISCONNECTED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The source with source_type=DISCONNECTED cannot be created",
        )

    # Validate source connectivity before creating
    is_reachable = await configuration_service.validate_source_connectivity(validated_source)
    if not is_reachable:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The source could not be reached. Please verify the configuration is correct.",
        )

    try:
        return await configuration_service.create_source(validated_source)
    except ResourceAlreadyExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.get(
    "",
    responses={
        status.HTTP_200_OK: {"description": "List of available source configurations", "model": SourceList},
    },
)
async def list_sources(
    project_id: Annotated[UUID, Depends(get_project_id)],
    configuration_service: Annotated[ConfigurationService, Depends(get_configuration_service)],
    limit: Annotated[int, Depends(PaginationLimit())],
    offset: Annotated[int, Query(ge=0)] = 0,
) -> SourceList:
    """List the available sources"""
    return await configuration_service.list_sources(
        project_id=project_id,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{source_id}",
    responses={
        status.HTTP_200_OK: {"description": "Source found", "model": Source},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid source ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Source not found"},
    },
)
async def get_source(
    project_id: Annotated[UUID, Depends(get_project_id)],
    source_id: Annotated[UUID, Depends(get_source_id)],
    configuration_service: Annotated[ConfigurationService, Depends(get_configuration_service)],
) -> Source:
    """Get info about a source"""
    try:
        return await configuration_service.get_source_by_id(source_id, project_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.patch(
    "/{source_id}",
    responses={
        status.HTTP_200_OK: {"description": "Source successfully updated", "model": Source},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid source ID or request body"},
        status.HTTP_404_NOT_FOUND: {"description": "Source not found"},
    },
)
async def update_source(
    project_id: Annotated[UUID, Depends(get_project_id)],
    source_id: Annotated[UUID, Depends(get_source_id)],
    source_config: Annotated[
        dict,
        Body(
            description=UPDATE_SOURCE_BODY_DESCRIPTION,
            openapi_examples=UPDATE_SOURCE_BODY_EXAMPLES,
        ),
    ],
    configuration_service: Annotated[ConfigurationService, Depends(get_configuration_service)],
) -> Source:
    """Reconfigure an existing source"""
    if "source_type" in source_config:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The 'source_type' field cannot be changed")
    try:
        return await configuration_service.update_source(source_id, project_id, source_config)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post(
    "/{source_id}:export",
    response_class=FileResponse,
    responses={
        status.HTTP_200_OK: {
            "description": "Source configuration exported as a YAML file",
            "content": {
                "application/x-yaml": {"schema": {"type": "string", "format": "binary"}},
            },
        },
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid source ID or request body"},
        status.HTTP_404_NOT_FOUND: {"description": "Source not found"},
    },
)
async def export_source(
    project_id: Annotated[UUID, Depends(get_project_id)],
    source_id: Annotated[UUID, Depends(get_source_id)],
    configuration_service: Annotated[ConfigurationService, Depends(get_configuration_service)],
) -> Response:
    """Export a source to file"""
    source = await configuration_service.get_source_by_id(source_id, project_id)
    if not source:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Source with ID {source_id} not found")

    yaml_content = yaml.safe_dump(source.model_dump(mode="json", exclude={"id", "project_id"}))

    return Response(
        content=yaml_content.encode("utf8"),
        media_type="application/x-yaml",
        headers={"Content-Disposition": f"attachment; filename=source_{source_id}.yaml"},
    )


@router.post(
    ":import",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {"description": "Source imported successfully", "model": Source},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid YAML format or source type is DISCONNECTED"},
    },
)
async def import_source(
    project_id: Annotated[UUID, Depends(get_project_id)],
    yaml_file: Annotated[UploadFile, File(description="YAML file containing the source configuration")],
    configuration_service: Annotated[ConfigurationService, Depends(get_configuration_service)],
) -> Source:
    """Import a source from file"""
    try:
        yaml_content = await yaml_file.read()
        source_data = yaml.safe_load(yaml_content)

        # Inject project_id from URL path
        source_data["project_id"] = str(project_id)

        source_config = SourceAdapter.validate_python(source_data)
        if source_config.source_type == SourceType.DISCONNECTED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The source with source_type=DISCONNECTED cannot be imported",
            )
        return await configuration_service.create_source(source_config)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid YAML format: {str(e)}")


@router.delete(
    "/{source_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {
            "description": "Source configuration successfully deleted",
        },
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid source ID or source is used by at least one pipeline"},
        status.HTTP_404_NOT_FOUND: {"description": "Source not found"},
        status.HTTP_409_CONFLICT: {"description": "Source is used by at least one pipeline"},
    },
)
async def delete_source(
    project_id: Annotated[UUID, Depends(get_project_id)],
    source_id: Annotated[UUID, Depends(get_source_id)],
    configuration_service: Annotated[ConfigurationService, Depends(get_configuration_service)],
) -> None:
    """Remove a source"""
    try:
        await configuration_service.delete_source_by_id(source_id, project_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ResourceInUseError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
