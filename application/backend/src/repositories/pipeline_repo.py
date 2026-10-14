# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from uuid import UUID

from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import PipelineDB
from pydantic_models import Pipeline
from repositories.base import BaseRepository
from repositories.mappers.pipeline_mapper import PipelineMapper


class PipelineRepository(BaseRepository):
    """Repository for pipeline-related database operations."""

    def __init__(self, db: AsyncSession):
        super().__init__(db, schema=PipelineDB)

    @property
    def to_schema(self) -> Callable[[Pipeline], PipelineDB]:
        return PipelineMapper.to_schema

    @property
    def from_schema(self) -> Callable[[PipelineDB], Pipeline]:
        return PipelineMapper.from_schema

    async def get_by_id(self, project_id: str | UUID) -> Pipeline | None:
        return await self.get_one(extra_filters={"project_id": self._id_to_str(project_id)})

    async def get_active_pipeline(self) -> Pipeline | None:
        """Get the active pipeline from database."""
        return await self.get_one(expressions=[PipelineDB.is_active])
