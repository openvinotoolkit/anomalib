# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Endpoints for managing pipelines"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, status
from fastapi.exceptions import HTTPException
from fastapi.openapi.models import Example
from pydantic import ValidationError

from api.dependencies import get_pipeline_metrics_service, get_pipeline_service, get_project_id
from pydantic_models.metrics import PipelineMetrics
from pydantic_models.pipeline import Pipeline, PipelineStatus
from services import PipelineService
from services.pipeline_metrics_service import PipelineMetricsService

router = APIRouter(prefix="/api/projects/{project_id}/pipeline", tags=["Pipelines"])

UPDATE_PIPELINE_BODY_DESCRIPTION = """
Partial pipeline configuration update. May contain any subset of fields including 'name', 'source_id',
'sink_id', or 'model_id'. Fields not included in the request will remain unchanged.
"""
UPDATE_PIPELINE_BODY_EXAMPLES = {
    "switch_model": Example(
        summary="Switch active model",
        description="Change the active model for the pipeline",
        value={
            "model_id": "c1feaabc-da2b-442e-9b3e-55c11c2c2ff3",
        },
    ),
    "reconfigure": Example(
        summary="Reconfigure pipeline",
        description="Change the name, source and sink of the pipeline",
        value={
            "name": "Updated Production Pipeline",
            "source_id": "e3cbd8d0-17b8-463e-85a2-4aaed031674e",
            "sink_id": "c6787c06-964b-4097-8eca-238b8cf79fc9",
        },
    ),
}


@router.get(
    "",
    responses={
        status.HTTP_200_OK: {"description": "Pipeline found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Pipeline not found"},
    },
    response_model_exclude_none=True,
)
async def get_pipeline(
    project_id: Annotated[UUID, Depends(get_project_id)],
    pipeline_service: Annotated[PipelineService, Depends(get_pipeline_service)],
) -> Pipeline:
    """Get info about a given pipeline"""
    return await pipeline_service.get_pipeline_by_id(project_id)


@router.patch(
    "",
    responses={
        status.HTTP_200_OK: {"description": "Pipeline successfully reconfigured"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID or request body"},
        status.HTTP_404_NOT_FOUND: {"description": "Pipeline not found"},
        status.HTTP_409_CONFLICT: {"description": "Pipeline cannot be reconfigured"},
    },
    response_model_exclude_none=True,
)
async def update_pipeline(
    project_id: Annotated[UUID, Depends(get_project_id)],
    pipeline_config: Annotated[
        dict,
        Body(
            description=UPDATE_PIPELINE_BODY_DESCRIPTION,
            openapi_examples=UPDATE_PIPELINE_BODY_EXAMPLES,
        ),
    ],
    pipeline_service: Annotated[PipelineService, Depends(get_pipeline_service)],
) -> Pipeline:
    """Reconfigure an existing pipeline"""
    if "status" in pipeline_config:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The 'status' field cannot be changed")
    try:
        return await pipeline_service.update_pipeline(project_id, pipeline_config)
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.post(
    ":run",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {
            "description": "Pipeline successfully set to run (also activates it if not already active).",
        },
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Pipeline not found"},
        status.HTTP_409_CONFLICT: {"description": "Pipeline cannot be run"},
    },
)
async def run_pipeline(
    project_id: Annotated[UUID, Depends(get_project_id)],
    pipeline_service: Annotated[PipelineService, Depends(get_pipeline_service)],
) -> None:
    """
    Set the pipeline status to RUNNING.
    The pipeline will start processing data from the source, run it through the model, and send results to the sink.
    If the pipeline is not yet activated, it will be activated as well.
    """
    try:
        await pipeline_service.activate_pipeline(project_id, set_running=True)
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.post(
    ":stop",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Pipeline stopped successfully."},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Pipeline not found"},
        status.HTTP_409_CONFLICT: {"description": "Pipeline cannot be stopped"},
    },
)
async def stop_pipeline(
    project_id: Annotated[UUID, Depends(get_project_id)],
    pipeline_service: Annotated[PipelineService, Depends(get_pipeline_service)],
) -> None:
    """
    Stops the running pipeline but does not deactivate it.
    This keeps the pipeline activated, switching to source passthrough mode.
    """
    active_pipeline = await pipeline_service.get_active_pipeline()
    if active_pipeline and active_pipeline.project_id != project_id:
        reason = f"another pipeline `{active_pipeline.project_id}` is currently {active_pipeline.status}."
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot stop pipeline `{project_id}`: {reason}",
        )
    if not (active_pipeline and active_pipeline.project_id == project_id and active_pipeline.status.is_running):
        reason = "the pipeline is not running."
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot stop pipeline `{project_id}`: {reason}",
        )
    try:
        await pipeline_service.update_pipeline(project_id, {"status": PipelineStatus.ACTIVE})
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.post(
    ":activate",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {
            "description": "Pipeline successfully activated (but not necessarily running yet).",
        },
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Pipeline not found"},
        status.HTTP_409_CONFLICT: {"description": "Pipeline cannot be activated"},
    },
)
async def activate_pipeline(
    project_id: Annotated[UUID, Depends(get_project_id)],
    pipeline_service: Annotated[PipelineService, Depends(get_pipeline_service)],
) -> None:
    """
    Set the pipeline status to ACTIVE.
    The pipeline will be ready to run but will not start processing data until set to RUNNING.
    """
    try:
        await pipeline_service.activate_pipeline(project_id)
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.post(
    ":disable",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Pipeline successfully disabled"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Pipeline not found"},
    },
)
async def disable_pipeline(
    project_id: Annotated[UUID, Depends(get_project_id)],
    pipeline_service: Annotated[PipelineService, Depends(get_pipeline_service)],
) -> None:
    """Stop a pipeline. The pipeline will become idle, and it won't process any data until re-enabled."""
    await pipeline_service.update_pipeline(project_id, {"status": PipelineStatus.IDLE})


@router.get(
    "/metrics",
    responses={
        status.HTTP_200_OK: {"description": "Pipeline metrics successfully calculated"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid project ID or duration parameter"},
        status.HTTP_404_NOT_FOUND: {"description": "Pipeline not found"},
    },
)
async def get_project_metrics(
    project_id: Annotated[UUID, Depends(get_project_id)],
    pipeline_metrics_service: Annotated[PipelineMetricsService, Depends(get_pipeline_metrics_service)],
    time_window: int = 60,
) -> PipelineMetrics:
    """
    Calculate model metrics for a pipeline over a specified time window.

    Returns inference latency metrics including average, min, max, 95th percentile,
    and latest latency measurements over the specified duration.
    """
    if time_window <= 0 or time_window > 3600:  # Limit to 1 hour max
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Duration must be between 1 and 3600 seconds",
        )

    try:
        return await pipeline_metrics_service.get_pipeline_metrics(project_id, time_window)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
