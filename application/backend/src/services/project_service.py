# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from loguru import logger

from db import get_async_db_session_ctx
from pydantic_models import Project, ProjectList, ProjectUpdate
from pydantic_models.base import Pagination
from repositories import ProjectRepository
from services import ConfigurationService
from services.dataset_snapshot_service import DatasetSnapshotService
from services.job_service import JobService
from services.media_service import MediaService
from services.model_service import ModelService
from services.pipeline_service import PipelineService
from services.video_service import VideoService


class ProjectService:
    @staticmethod
    async def get_project_list(limit: int, offset: int) -> ProjectList:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRepository(session)
            total = await repo.get_all_count()
            items = await repo.get_all_pagination(limit=limit, offset=offset)
        return ProjectList(
            projects=items,
            pagination=Pagination(
                limit=limit,
                offset=offset,
                count=len(items),
                total=total,
            ),
        )

    @staticmethod
    async def get_project_by_id(project_id: UUID) -> Project | None:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRepository(session)
            return await repo.get_by_id(project_id)

    @staticmethod
    async def create_project(project: Project) -> Project:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRepository(session)
            return await repo.save(project)

    @staticmethod
    async def update_project(project_id: UUID, project_update: ProjectUpdate) -> Project | None:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRepository(session)
            project = await repo.get_by_id(project_id)
            if project is None:
                return None
            return await repo.update(project, project_update.model_dump(exclude_unset=True))

    @staticmethod
    @logger.catch(reraise=True)
    async def delete_project(project_id: UUID) -> None:
        """
        Delete a project in a transactional manner.

        Phase 1: Database Deletion
        - Deletes all related DB records in a single transaction.
        - Rolls back if any deletion fails.

        Phase 2: File Cleanup
        - Performed only after successful DB transaction.
        - Best-effort deletion of project folders.
        - Logs warnings on failure but does not raise exceptions.

        Args:
            project_id: UUID of the project to delete.
        """
        # Phase 1: Database Deletion (Transaction)
        try:
            async with get_async_db_session_ctx() as session:
                # 1. Pipelines
                await PipelineService.delete_project_pipelines_db(session, project_id, commit=False)

                # 2. Models
                await ModelService.delete_project_models_db(session, project_id, commit=False)

                # 3. Dataset Snapshots
                await DatasetSnapshotService.delete_project_snapshots_db(session, project_id, commit=False)

                # 4. Media
                await MediaService.delete_project_media_db(session, project_id, commit=False)

                # 5. Jobs
                await JobService.delete_project_jobs_db(session, project_id, commit=False)

                # 6. Sources
                await ConfigurationService.delete_project_source_db(session, project_id, commit=False)

                # 7. Sinks
                await ConfigurationService.delete_project_sink_db(session, project_id, commit=False)

                # 8. Project
                project_repo = ProjectRepository(session)
                await project_repo.delete_by_id(project_id)

                # Commit transaction
                await session.commit()
        except Exception as e:
            # Rollback is automatic on exception due to context manager
            raise RuntimeError(f"Failed to delete project {project_id} from database: {e}")

        logger.info(f"Deleted project {project_id} from database, proceeding to file cleanup.")

        # Phase 2: File Cleanup (Best Effort)
        # Only reached if DB transaction succeeds

        # Cleanup Models
        await ModelService.cleanup_project_model_files(project_id)

        # Cleanup Snapshots
        await DatasetSnapshotService.cleanup_project_snapshot_files(project_id)

        # Cleanup Media
        await MediaService.cleanup_project_media_files(project_id)

        # Cleanup Videos
        await VideoService.cleanup_project_videos(project_id)

        logger.success(f"Deleted project {project_id} from database and filesystem.")
