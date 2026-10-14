# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
import queue
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from api.dependencies import get_media_service, get_project_id, get_scheduler
from api.endpoints import API_PREFIX
from core import Scheduler
from pydantic_models import Media
from services.media_service import MediaService

capture_api_prefix_url = API_PREFIX + "/projects/{project_id}"
router = APIRouter(
    prefix=capture_api_prefix_url,
    tags=["capture"],
)


@router.get(
    "/capture",
    response_model_exclude_none=True,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {"description": "Image captured successfully"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid image file"},
        status.HTTP_404_NOT_FOUND: {"description": "Source or project not found"},
    },
)
async def capture(
    media_service: Annotated[MediaService, Depends(get_media_service)],
    project_id: Annotated[UUID, Depends(get_project_id)],
    scheduler: Annotated[Scheduler, Depends(get_scheduler)],
    background_tasks: BackgroundTasks,
) -> Media:
    """Endpoint to capture an image"""
    try:
        stream_data = await asyncio.to_thread(lambda: scheduler.frame_queue.get(timeout=1))
    except queue.Empty as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No frame available to capture. Please make sure the stream is active.",
        ) from e

    media = await media_service.upload_image(
        project_id=project_id,
        image=stream_data.frame_data,
        is_anomalous=False,
        extension=".png",
    )

    background_tasks.add_task(
        media_service.generate_thumbnail,
        project_id=project_id,
        media_id=media.id,
        image=stream_data.frame_data,
    )
    return media
