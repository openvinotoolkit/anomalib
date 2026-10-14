# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from uuid import UUID

from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import JobDB
from pydantic_models import Job
from pydantic_models.job import JobStatus, JobType, TrainJobPayload
from repositories.base import BaseRepository
from repositories.mappers import JobMapper


class JobRepository(BaseRepository):
    def __init__(self, db: AsyncSession):
        super().__init__(db, schema=JobDB)

    @property
    def to_schema(self) -> Callable[[Job], JobDB]:
        return JobMapper.to_schema

    @property
    def from_schema(self) -> Callable[[JobDB], Job]:
        return JobMapper.from_schema

    async def is_job_duplicate(self, project_id: UUID, payload: TrainJobPayload) -> bool:
        # Convert payload to dict for comparison
        payload_dict = payload.model_dump()

        # Check for jobs with same payload that are not completed
        existing_job = await self.get_one(
            extra_filters={"project_id": self._id_to_str(project_id), "payload": payload_dict},
            expressions=[
                JobDB.status != JobStatus.COMPLETED,
                JobDB.status != JobStatus.FAILED,
                JobDB.status != JobStatus.CANCELED,
            ],
        )

        return existing_job is not None

    async def get_pending_job_by_type(self, job_type: JobType) -> Job | None:
        return await self.get_one(
            extra_filters={"type": job_type, "status": JobStatus.PENDING},
            order_by=self.schema.created_at,
            ascending=True,
        )
