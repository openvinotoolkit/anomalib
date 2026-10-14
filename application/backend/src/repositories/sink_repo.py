# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from uuid import UUID

from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import SinkDB
from pydantic_models import Sink
from repositories.base import ProjectBaseRepository
from repositories.mappers import SinkMapper


class SinkRepository(ProjectBaseRepository):
    """Repository for sink-related database operations."""

    def __init__(self, db: AsyncSession, project_id: UUID):
        super().__init__(db, schema=SinkDB, project_id=project_id)

    @property
    def to_schema(self) -> Callable[[Sink], SinkDB]:
        return SinkMapper.to_schema

    @property
    def from_schema(self) -> Callable[[SinkDB], Sink]:
        return SinkMapper.from_schema
