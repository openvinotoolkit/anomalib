# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
import datetime
import json
import logging
from collections.abc import AsyncGenerator
from uuid import UUID

import anyio
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio.session import AsyncSession
from sse_starlette import ServerSentEvent

from db import get_async_db_session_ctx
from exceptions import DuplicateJobException, JobNotDeletableException, ResourceNotFoundException
from pydantic_models import Job, JobList, JobType
from pydantic_models.base import Pagination
from pydantic_models.job import JobCancelled, JobStatus, JobSubmitted, TrainJobPayload
from repositories import JobRepository

logger = logging.getLogger(__name__)


class JobService:
    @staticmethod
    async def get_job_list(limit: int, offset: int, extra_filters: dict | None = None) -> JobList:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            total = await repo.get_all_count(extra_filters=extra_filters)
            items = await repo.get_all_pagination(extra_filters=extra_filters, limit=limit, offset=offset)
        return JobList(
            jobs=items,
            pagination=Pagination(
                limit=limit,
                offset=offset,
                count=len(items),
                total=total,
            ),
        )

    @staticmethod
    async def get_job_list_streaming(extra_filters: dict | None = None) -> AsyncGenerator[Job]:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            async for item in repo.get_all_streaming(extra_filters=extra_filters):
                yield item  # need to re-yield to extends session's context

    @staticmethod
    async def get_job_by_id(job_id: UUID | str) -> Job | None:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            return await repo.get_by_id(job_id)

    @staticmethod
    async def submit_train_job(payload: TrainJobPayload) -> JobSubmitted:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            if await repo.is_job_duplicate(project_id=payload.project_id, payload=payload):
                raise DuplicateJobException

            try:
                job = Job(
                    project_id=payload.project_id,
                    type=JobType.TRAINING,
                    payload=payload.model_dump(),
                    message="Training job submitted",
                )
                saved_job = await repo.save(job)
                return JobSubmitted(job_id=saved_job.id)
            except IntegrityError:
                raise ResourceNotFoundException(resource_id=payload.project_id, resource_name="project")

    @staticmethod
    async def get_pending_train_job() -> Job | None:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            return await repo.get_pending_job_by_type(JobType.TRAINING)

    @staticmethod
    async def update_job_status(
        job_id: UUID,
        status: JobStatus,
        message: str | None = None,
        progress: int | None = None,
    ) -> None:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            job = await repo.get_by_id(job_id)
            if job is None:
                raise ResourceNotFoundException(resource_id=job_id, resource_name="job")
            updates: dict = {"status": status}
            if message is not None:
                updates["message"] = message
            progress_ = 100 if status is JobStatus.COMPLETED else progress

            if status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELED}:
                updates["end_time"] = datetime.datetime.now(tz=datetime.UTC)

            if progress_ is not None:
                updates["progress"] = progress_
            await repo.update(job, updates)

    @staticmethod
    async def update_job_progress(
        job_id: UUID,
        progress: int,
        message: str | None = None,
    ) -> None:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            job = await repo.get_by_id(job_id)
            if job is None:
                raise ResourceNotFoundException(resource_id=job_id, resource_name="job")
            updates: dict = {"progress": progress}
            if message is not None:
                updates["message"] = message
            await repo.update(job, updates)

    @classmethod
    async def is_job_still_running(cls, job_id: UUID | str) -> bool:
        job = await cls.get_job_by_id(job_id=job_id)
        if job is None:
            raise ResourceNotFoundException(resource_id=job_id, resource_name="job")
        return job.status == JobStatus.RUNNING

    @classmethod
    async def stream_logs(cls, job_id: UUID | str) -> AsyncGenerator[ServerSentEvent]:
        from core.logging.utils import get_job_logs_path  # noqa: PLC0415

        log_file = get_job_logs_path(job_id=job_id)
        if not await anyio.Path(log_file).exists():
            raise ResourceNotFoundException(resource_id=job_id, resource_name="job_logs")

        # Cache job status and only check every 2 seconds
        status_check_interval = 2.0  # seconds
        last_status_check = 0.0
        cached_still_running = True
        loop = asyncio.get_running_loop()

        async with await anyio.open_file(log_file, encoding="utf-8") as f:
            while True:
                line = await f.readline()
                now = loop.time()
                # Only check job status every status_check_interval seconds
                if now - last_status_check > status_check_interval:
                    cached_still_running = await cls.is_job_still_running(job_id=job_id)
                    last_status_check = now
                still_running = cached_still_running
                if not line:
                    # wait for more lines if job is still running
                    if still_running:
                        await asyncio.sleep(0.5)
                        continue
                    # No more lines are expected
                    else:
                        break
                yield ServerSentEvent(data=line.rstrip())

    @classmethod
    async def stream_progress(cls, job_id: UUID | str) -> AsyncGenerator[ServerSentEvent]:
        """Stream the progress of a job by its ID"""
        still_running = True
        while still_running:
            job = await cls.get_job_by_id(job_id=job_id)
            if job is None:
                raise ResourceNotFoundException(resource_id=job_id, resource_name="job")
            yield ServerSentEvent(data=json.dumps({"progress": job.progress, "message": job.message}))
            still_running = job.status in {JobStatus.RUNNING, JobStatus.PENDING}
            await asyncio.sleep(0.5)

    @classmethod
    async def cancel_job(cls, job_id: UUID | str) -> JobCancelled:
        """Cancel a job by its ID"""
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            job = await repo.get_by_id(job_id)
            if job is None:
                raise ResourceNotFoundException(resource_id=job_id, resource_name="job")

            await repo.update(job, {"status": JobStatus.CANCELED})
            return JobCancelled(job_id=job.id)

    @classmethod
    async def delete_job(cls, job_id: UUID | str) -> None:
        """Delete a failed or canceled job by its ID."""
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            job = await repo.get_by_id(job_id)
            if job is None:
                raise ResourceNotFoundException(resource_id=job_id, resource_name="job")
            if job.status not in {JobStatus.FAILED, JobStatus.CANCELED}:
                raise JobNotDeletableException(job_id=job_id, job_status=job.status)
            await repo.delete_by_id(job_id)

    @classmethod
    async def delete_project_jobs_db(cls, session: AsyncSession, project_id: UUID, commit: bool = False) -> None:
        """Delete all jobs associated with a project from the database."""
        repo = JobRepository(session)
        await repo.delete_all(commit=commit, extra_filters={"project_id": str(project_id)})

    @staticmethod
    async def has_running_jobs(project_id: str | UUID) -> bool:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            count = await repo.get_all_count(
                extra_filters={"status": JobStatus.RUNNING, "project_id": str(project_id)},
            )
            return count > 0
