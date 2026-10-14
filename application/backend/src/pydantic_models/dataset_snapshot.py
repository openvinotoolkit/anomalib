# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from pydantic_models.base import BaseIDModel, Pagination


class DatasetSnapshot(BaseIDModel):
    """
    Pydantic model for DatasetSnapshot.
    """

    project_id: UUID
    filename: str
    count: int
    created_at: datetime | None = Field(default=None)

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "76e07d18-196e-4e33-bf98-ac1d35dca4cb",
                "project_id": "16e07d18-196e-4e33-bf98-ac1d35dcaaaa",
                "filename": "dataset_snapshot_123.parquet",
                "count": 42,
                "created_at": "2025-01-01T12:00:00",
            },
        },
    }


class DatasetSnapshotList(BaseModel):
    snapshots: list[DatasetSnapshot]
    pagination: Pagination
