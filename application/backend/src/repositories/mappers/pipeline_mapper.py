# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from db.schema import PipelineDB
from pydantic_models import Pipeline, PipelineStatus
from repositories.mappers import ModelMapper, SinkMapper, SourceMapper
from repositories.mappers.base_mapper_interface import IBaseMapper


class PipelineMapper(IBaseMapper):
    """Mapper for Pipeline schema entity <-> DB entity conversions."""

    @staticmethod
    def from_schema(pipeline_db: PipelineDB) -> Pipeline:
        """Convert Pipeline db schema to Pipeline pydantic model."""
        return Pipeline(
            project_id=UUID(pipeline_db.project_id),
            source=SourceMapper.from_schema(pipeline_db.source) if pipeline_db.source else None,
            sink=SinkMapper.from_schema(pipeline_db.sink) if pipeline_db.sink else None,
            model=ModelMapper.from_schema(pipeline_db.model) if pipeline_db.model else None,
            sink_id=UUID(pipeline_db.sink_id) if pipeline_db.sink_id else None,
            model_id=UUID(pipeline_db.model_id) if pipeline_db.model_id else None,
            source_id=UUID(pipeline_db.source_id) if pipeline_db.source_id else None,
            status=PipelineStatus.from_bool(pipeline_db.is_running, pipeline_db.is_active),
            inference_device=pipeline_db.inference_device.upper(),
            overlay=pipeline_db.overlay,
        )

    @staticmethod
    def to_schema(pipeline: Pipeline) -> PipelineDB:
        """Convert Pipeline pydantic model to db schema."""
        return PipelineDB(
            project_id=str(pipeline.project_id),
            source_id=str(pipeline.source_id) if pipeline.source_id else None,
            model_id=str(pipeline.model_id) if pipeline.model_id else None,
            sink_id=str(pipeline.sink_id) if pipeline.sink_id else None,
            is_running=pipeline.status.is_running,
            is_active=pipeline.status.is_active,
            inference_device=pipeline.inference_device.upper(),
            overlay=pipeline.overlay,
        )
