# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status

from api.dependencies import PaginationLimit, get_job_service, get_pipeline_service, get_project_id, get_project_service
from api.endpoints import API_PREFIX
from pydantic_models import Project, ProjectList, ProjectUpdate
from services import JobService, PipelineService, ProjectService

project_api_prefix_url = API_PREFIX + "/projects"
project_router = APIRouter(
    prefix=project_api_prefix_url,
    tags=["Project"],
)


@project_router.get("")
async def get_projects(
    project_service: Annotated[ProjectService, Depends(get_project_service)],
    limit: Annotated[int, Depends(PaginationLimit())],
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ProjectList:
    """Endpoint to get list of all projects"""
    return await project_service.get_project_list(limit=limit, offset=offset)


@project_router.post("")
async def create_project(
    project_service: Annotated[ProjectService, Depends(get_project_service)],
    project: Annotated[Project, Body()],
) -> Project:
    """Endpoint to create a new project"""
    return await project_service.create_project(project)


@project_router.get("/{project_id}")
async def get_project_by_id(
    project_service: Annotated[ProjectService, Depends(get_project_service)],
    project_id: Annotated[UUID, Depends(get_project_id)],
) -> Project:
    """Endpoint to get project metadata by ID"""
    project = await project_service.get_project_by_id(project_id)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return project


@project_router.patch("/{project_id}")
async def update_project(
    project_service: Annotated[ProjectService, Depends(get_project_service)],
    project_id: Annotated[UUID, Depends(get_project_id)],
    project_update: Annotated[ProjectUpdate, Body()],
) -> Project:
    """Endpoint to update project metadata by ID"""
    project = await project_service.update_project(project_id, project_update)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return project


@project_router.delete("/{project_id}")
async def delete_project(
    job_service: Annotated[JobService, Depends(get_job_service)],
    pipeline_service: Annotated[PipelineService, Depends(get_pipeline_service)],
    project_service: Annotated[ProjectService, Depends(get_project_service)],
    project_id: Annotated[UUID, Depends(get_project_id)],
) -> None:
    """Endpoint to delete a project by ID"""
    project = await project_service.get_project_by_id(project_id)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    active_pipeline = await pipeline_service.get_active_pipeline()
    if active_pipeline and active_pipeline.project_id == project_id and active_pipeline.is_running:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot delete project with active pipeline. Please deactivate the pipeline first.",
        )
    if await job_service.has_running_jobs(project_id=project_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot delete project with running jobs. Please cancel the jobs first.",
        )
    try:
        await project_service.delete_project(project_id)
    except RuntimeError as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete project. Deletion rolled back. Error: {str(err)}",
        )
