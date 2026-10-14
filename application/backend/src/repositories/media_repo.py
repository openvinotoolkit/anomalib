# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from uuid import UUID

from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import MediaDB
from pydantic_models import Media
from repositories.base import ProjectBaseRepository
from repositories.mappers import MediaMapper


class MediaRepository(ProjectBaseRepository):
    def __init__(self, db: AsyncSession, project_id: str | UUID):
        super().__init__(db, schema=MediaDB, project_id=str(project_id))

    @property
    def to_schema(self) -> Callable[[Media], MediaDB]:
        return MediaMapper.to_schema

    @property
    def from_schema(self) -> Callable[[MediaDB], Media]:
        return MediaMapper.from_schema
