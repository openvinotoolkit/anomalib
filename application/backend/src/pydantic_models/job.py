# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import re
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from loguru import logger
from pydantic import BaseModel, Field, ValidationInfo, computed_field, field_serializer, field_validator

from pydantic_models.base import BaseIDModel, Pagination
from pydantic_models.system import DeviceType


class JobType(StrEnum):
    TRAINING = "training"
    OPTIMIZATION = "optimization"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class Job(BaseIDModel):
    project_id: UUID
    type: JobType = JobType.TRAINING
    progress: int = Field(default=0, ge=0, le=100, description="Progress percentage from 0 to 100")
    status: JobStatus = JobStatus.PENDING
    payload: dict[str, Any]
    message: str = "Job created"
    start_time: datetime | None = None
    end_time: datetime | None = None

    @field_serializer("project_id")
    def serialize_project_id(self, project_id: UUID, _info: Any) -> str:
        return str(project_id)

    @property
    def is_running(self) -> bool:
        return self.status == JobStatus.RUNNING

    @property
    def has_failed(self) -> bool:
        return self.status == JobStatus.FAILED


class JobList(BaseModel):
    jobs: list[Job]
    pagination: Pagination


class JobSubmitted(BaseModel):
    job_id: UUID


class JobCancelled(BaseModel):
    job_id: UUID

    @computed_field
    def message(self) -> str:
        return f"Job with ID `{self.job_id}` marked as cancelled."


class TrainingDevice(BaseModel):
    """Device specification for training."""

    type: DeviceType = Field(..., description="Device type, e.g. 'cpu', 'xpu', 'cuda'")
    index: int | None = Field(default=None, ge=0, description="Device index (null for CPU/MPS/NPU)")

    @field_validator("index")
    @classmethod
    def validate_index_for_device_type(cls, index: int | None, info: ValidationInfo) -> int | None:
        device_type = info.data.get("type") if info.data else None
        # Normalize device_type to str if given as DeviceType enum
        device_type_str = str(device_type).lower() if device_type is not None else ""
        indexed_types = {"cuda", "xpu"}
        non_indexed_types = {"cpu", "mps", "npu"}

        if device_type_str in non_indexed_types:
            if index is not None:
                msg = f"Device type '{device_type}' does not support an index. Got index={index}. Disregarding index."
                logger.warning(msg)
                index = None
        elif device_type_str in indexed_types and index is None:
            msg = (
                f"Device type '{device_type_str}' requires an index (e.g., device 'cuda:0', 'xpu:0')."
                " Using default index 0."
            )
            logger.warning(msg)
            index = 0
        # For unknown device types, allow both options for forward compatibility
        return index


MAX_MODEL_NAME_LENGTH = 64
MODEL_NAME_PATTERN = re.compile(r"^[a-z0-9_]+$")


class TrainJobPayload(BaseModel):
    project_id: UUID = Field(exclude=True)
    model_name: str
    device: TrainingDevice | None = Field(default=None)
    dataset_snapshot_id: str | None = Field(default=None)  # used because UUID is not JSON serializable
    max_epochs: int | None = Field(default=None, ge=1, le=10000)

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, model_name: str) -> str:
        # Normalize to lowercase so identifiers are case-insensitive and canonical on disk.
        normalized = model_name.strip().lower()
        if not normalized:
            raise ValueError("model_name must not be empty")
        if len(normalized) > MAX_MODEL_NAME_LENGTH:
            raise ValueError(
                f"model_name must be at most {MAX_MODEL_NAME_LENGTH} characters long, got {len(normalized)} characters",
            )
        if not MODEL_NAME_PATTERN.fullmatch(normalized):
            raise ValueError(
                "model_name may only contain lowercase letters, digits, and underscores "
                f"(matching [a-z0-9_]+); got '{model_name}'",
            )
        return normalized
