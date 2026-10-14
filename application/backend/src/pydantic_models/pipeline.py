# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from functools import cached_property
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pydantic_models.model import Model
from pydantic_models.sink import Sink
from pydantic_models.source import Source


class PipelineStatus(StrEnum):
    IDLE = "idle"
    ACTIVE = "active"
    RUNNING = "running"

    @classmethod
    def from_bool(cls, is_running: bool, is_active: bool) -> "PipelineStatus":
        if is_running:
            return cls.RUNNING
        if is_active:
            return cls.ACTIVE
        return cls.IDLE

    @property
    def is_running(self) -> bool:
        return self == PipelineStatus.RUNNING

    @property
    def is_active(self) -> bool:
        return self in {PipelineStatus.RUNNING, PipelineStatus.ACTIVE}


class Pipeline(BaseModel):
    project_id: UUID  # ID of the project this pipeline belongs to
    source: Source | None = None  # None if disconnected
    sink: Sink | None = None  # None if disconnected
    model: Model | None = None  # None if no model is selected
    source_id: UUID | None = Field(
        default=None,
        exclude=True,
    )  # ID of the source, used for DB mapping, not exposed in API
    sink_id: UUID | None = Field(default=None, exclude=True)  # ID of the sink, used for DB mapping, not exposed in API
    model_id: UUID | None = Field(
        default=None,
        exclude=True,
    )  # ID of the model, used for DB mapping, not exposed in API
    status: PipelineStatus = PipelineStatus.IDLE  # Current status of the pipeline
    inference_device: str = Field(default="CPU")
    overlay: bool | None = Field(default=None)

    # TODO: can be confused with status.is_running / is_active, consider refactoring
    is_running: bool | None = Field(default=None, exclude=True)  # If set will overwrite status
    is_active: bool | None = Field(default=None, exclude=True)  # If set will overwrite status

    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "project_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "source": {
                    "source_type": "video_file",
                    "name": "Sample Video",
                    "id": "712750b2-5a82-47ee-8fba-f3dc96cb615d",
                    "video_path": "/path/to/video.mp4",
                },
                "sink": {
                    "id": "b5787c06-964b-4097-8eca-238b8cf79fc8",
                    "sink_type": "folder",
                    "name": "Local Folder",
                    "folder_path": "/path/to/output",
                    "output_formats": ["image_original", "image_with_predictions", "predictions"],
                    "rate_limit": 0.2,
                },
                "model": {
                    "id": "76e07d18-196e-4e33-bf98-ac1d35dca4cb",
                    "name": "PatchCore",
                    "format": "openvino",
                },
                "status": "running",
            },
        },
    )

    @model_validator(mode="after")
    def validate_running_status(self) -> "Pipeline":
        if self.is_running:
            self.status = PipelineStatus.RUNNING
        if self.status == PipelineStatus.RUNNING and any(
            x is None for x in (self.source_id or self.source, self.model_id or self.model)
        ):
            raise ValueError("Pipeline cannot be in 'running' status when source, or model is not configured.")
        return self

    @cached_property
    def id(self) -> UUID:
        return self.project_id
