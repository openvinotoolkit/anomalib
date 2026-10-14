# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from multiprocessing.synchronize import Condition
from uuid import UUID

from loguru import logger
from sqlalchemy.ext.asyncio.session import AsyncSession

from db import get_async_db_session_ctx
from pydantic_models import Pipeline, PipelineStatus
from repositories import PipelineRepository
from services import ActivePipelineConflictError, ActivePipelineService, ResourceNotFoundError
from services.exceptions import ResourceType
from services.model_service import ModelService

MSG_ERR_DELETE_RUNNING_PIPELINE = "Cannot delete a running pipeline."


class PipelineService:
    def __init__(
        self,
        active_pipeline_service: ActivePipelineService,
        config_changed_condition: Condition,
        model_service: ModelService,
    ) -> None:
        self._active_pipeline_service: ActivePipelineService = active_pipeline_service
        self._config_changed_condition: Condition = config_changed_condition
        self._model_service: ModelService = model_service

    def _notify_source_changed(self) -> None:
        with self._config_changed_condition:
            self._config_changed_condition.notify_all()

    async def _notify_sink_changed(self) -> None:
        await self._active_pipeline_service.reload()

    async def _notify_pipeline_changed(self) -> None:
        self._notify_source_changed()
        await self._notify_sink_changed()

    @staticmethod
    async def get_pipeline_by_id(project_id: UUID, session: AsyncSession | None = None) -> Pipeline:
        """Retrieve a pipeline by project ID."""
        if session is None:
            async with get_async_db_session_ctx() as db_session:
                repo = PipelineRepository(db_session)
                pipeline = await repo.get_by_id(project_id)
        else:
            repo = PipelineRepository(session)
            pipeline = await repo.get_by_id(project_id)
        if not pipeline:
            raise ResourceNotFoundError(resource_type=ResourceType.PIPELINE, resource_id=str(project_id))
        return pipeline

    async def update_pipeline(self, project_id: UUID, partial_config: dict) -> Pipeline:
        """Update an existing pipeline."""
        async with get_async_db_session_ctx() as session:
            pipeline = await self.get_pipeline_by_id(project_id, session)
            repo = PipelineRepository(session)
            updated = await repo.update(pipeline, partial_config)
            await session.commit()
            # notify source changes
            if pipeline.source != updated.source:
                self._notify_source_changed()
            if pipeline.status.is_running and updated.status.is_running:
                # source/sink changed or disconnected
                old_source_id = pipeline.source.id if pipeline.source else None
                new_source_id = updated.source.id if updated.source else None
                old_sink_id = pipeline.sink.id if pipeline.sink else None
                new_sink_id = updated.sink.id if updated.sink else None
                if old_source_id != new_source_id or old_sink_id != new_sink_id:
                    await self._notify_pipeline_changed()

                # If the active model changes while running, notify inference to reload
                if pipeline.model.id != updated.model.id:  # type: ignore[union-attr]
                    self._model_service.activate_model()
            elif pipeline.status != updated.status:
                # If the pipeline is being activated or stopped
                await self._notify_pipeline_changed()
                # Intentionally call activate_model on status change regardless of whether a model exists.
                self._model_service.activate_model()
            if updated.inference_device != pipeline.inference_device or updated.overlay != pipeline.overlay:
                # reload model on device change
                self._model_service.activate_model()
            return updated

    @staticmethod
    async def get_active_pipeline() -> Pipeline | None:
        """Retrieve the currently active (running) pipeline from the database."""
        async with get_async_db_session_ctx() as session:
            return await PipelineRepository(session).get_active_pipeline()

    async def activate_pipeline(self, project_id: UUID, set_running: bool = False) -> Pipeline:
        """Activate a pipeline. If set_running is True, set the pipeline status to RUNNING."""
        active_pipeline = await self.get_active_pipeline()
        if active_pipeline and active_pipeline.project_id != project_id:
            raise ActivePipelineConflictError(
                pipeline_id=str(project_id),
                reason=(
                    f"another pipeline is already active. "
                    f"Please disable the pipeline {active_pipeline.id} before activating a new one"
                ),
            )
        if active_pipeline and (
            (active_pipeline.status == PipelineStatus.ACTIVE and not set_running)
            or (active_pipeline.status == PipelineStatus.RUNNING and set_running)
        ):
            logger.info(
                f"Activating already {active_pipeline.status.value.lower()} pipeline `{active_pipeline.id}`, "
                f"no changes made.",
            )
            return active_pipeline
        new_status = PipelineStatus.RUNNING if set_running else PipelineStatus.ACTIVE
        return await self.update_pipeline(project_id, {"status": new_status})

    @classmethod
    async def delete_project_pipelines_db(cls, session: AsyncSession, project_id: UUID, commit: bool = False) -> None:
        """Delete all pipelines associated with a project from the database."""
        repo = PipelineRepository(session)
        await repo.delete_all(commit=commit, extra_filters={"project_id": str(project_id)})
