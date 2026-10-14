# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from datetime import datetime
from uuid import UUID

import sqlalchemy as sa
from loguru import logger
from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import PipelineDB, ProjectDB
from pydantic_models import Project
from repositories.base import BaseRepository
from repositories.mappers import ProjectMapper


class ProjectRepository(BaseRepository):
    def __init__(self, db: AsyncSession):
        super().__init__(db, ProjectDB)

    @property
    def to_schema(self) -> Callable[[Project], ProjectDB]:
        return ProjectMapper.to_schema

    @property
    def from_schema(self) -> Callable[[ProjectDB], Project]:
        return ProjectMapper.from_schema

    async def save(self, project: Project) -> Project:
        project_schema: ProjectDB = self.to_schema(project)
        project_schema.pipeline = PipelineDB(
            project_id=project_schema.id,
            inference_device="CPU",
        )
        self.db.add(project_schema)
        await self.db.commit()
        return project

    async def update_dataset_timestamp(self, project_id: str | UUID) -> None:
        """Update the dataset_updated_at timestamp for the given project."""
        await self.db.execute(
            sa.update(ProjectDB)
            .where(ProjectDB.id == str(project_id))
            .values(
                dataset_updated_at=sa.func.current_timestamp(),
                updated_at=sa.func.current_timestamp(),
            ),
        )
        logger.info(f"Updated dataset timestamp for project {project_id} to current time.")
        await self.db.commit()

    async def get_dataset_timestamp(self, project_id: str | UUID) -> datetime:
        """Get the dataset_updated_at timestamp for the given project."""
        result = await self.db.execute(sa.select(self.schema.dataset_updated_at).where(ProjectDB.id == str(project_id)))
        return result.scalar_one()
