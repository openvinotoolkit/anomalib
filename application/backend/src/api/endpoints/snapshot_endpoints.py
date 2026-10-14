# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse

from api.dependencies import PaginationLimit, get_project_id, get_snapshot_id
from api.endpoints import API_PREFIX
from pydantic_models.dataset_snapshot import DatasetSnapshot, DatasetSnapshotList
from repositories.binary_repo import DatasetSnapshotBinaryRepository
from services.dataset_snapshot_service import DatasetSnapshotService

router = APIRouter(
    prefix=API_PREFIX + "/projects/{project_id}",
    tags=["snapshots"],
)


@router.get(
    "/snapshots",
    responses={
        status.HTTP_200_OK: {"description": "List of available dataset snapshots"},
    },
    response_model_exclude_none=True,
)
async def get_snapshot_list(
    project_id: Annotated[UUID, Depends(get_project_id)],
    limit: Annotated[int, Depends(PaginationLimit())],
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DatasetSnapshotList:
    """Endpoint to get list of all snapshots"""
    return await DatasetSnapshotService.list_snapshots(project_id=project_id, limit=limit, offset=offset)


@router.get(
    "/snapshots/{snapshot_id}",
    response_model_exclude_none=True,
)
async def get_snapshot(
    project_id: Annotated[UUID, Depends(get_project_id)],
    snapshot_id: UUID,
) -> DatasetSnapshot:
    """Endpoint to get snapshot file by ID"""
    return await DatasetSnapshotService.get_snapshot(project_id=project_id, snapshot_id=snapshot_id)


@router.get(
    "/snapshots/{snapshot_id}/parquet",
    response_model_exclude_none=True,
    responses={
        status.HTTP_200_OK: {"description": "Snapshot Parquet file retrieved successfully"},
        status.HTTP_404_NOT_FOUND: {"description": "Snapshot not found"},
    },
)
async def get_snapshot_file(
    project_id: Annotated[UUID, Depends(get_project_id)],
    snapshot_id: Annotated[UUID, Depends(get_snapshot_id)],
) -> FileResponse:
    """Endpoint to get snapshot file by ID"""
    try:
        snapshot_bin_repo = DatasetSnapshotBinaryRepository(project_id=project_id)
        snapshot_path = snapshot_bin_repo.get_snapshot_path(snapshot_id)
        return FileResponse(snapshot_path)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Snapshot with ID {snapshot_id} not found",
        ) from e
