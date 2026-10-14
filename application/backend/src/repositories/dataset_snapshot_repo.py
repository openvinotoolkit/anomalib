# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import DatasetSnapshotDB
from pydantic_models.dataset_snapshot import DatasetSnapshot
from repositories.base import ProjectBaseRepository
from repositories.mappers.dataset_snapshot_mapper import DatasetSnapshotMapper


class DatasetSnapshotRepository(ProjectBaseRepository[DatasetSnapshot, DatasetSnapshotDB]):
    """Repository for Dataset Snapshot operations."""

    def __init__(self, db: AsyncSession, project_id: str | UUID):
        super().__init__(db, schema=DatasetSnapshotDB, project_id=project_id)

    @property
    def to_schema(self) -> Callable[[DatasetSnapshot], DatasetSnapshotDB]:
        return DatasetSnapshotMapper.to_schema

    @property
    def from_schema(self) -> Callable[[DatasetSnapshotDB], DatasetSnapshot]:
        return DatasetSnapshotMapper.from_schema

    async def get_latest_snapshot(self) -> DatasetSnapshot | None:
        """Get the latest dataset snapshot for the project."""
        result = await self.db.execute(
            sa.select(self.schema)
            .where(self.schema.project_id == self.project_id)
            .order_by(self.schema.created_at.desc())
            .limit(1),
        )
        snapshot_db = result.scalar_one_or_none()
        return self.from_schema(snapshot_db) if snapshot_db else None
