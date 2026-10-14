# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from uuid import uuid4

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class ProjectDB(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    dataset_updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())

    pipeline = relationship("PipelineDB", back_populates="project", uselist=False, lazy="joined")


class MediaDB(Base):
    __tablename__ = "media"

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    size: Mapped[int] = mapped_column(nullable=False)
    width: Mapped[int] = mapped_column(nullable=False)
    height: Mapped[int] = mapped_column(nullable=False)
    is_anomalous: Mapped[bool] = mapped_column(Boolean, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())


class DatasetSnapshotDB(Base):
    __tablename__ = "dataset_snapshot"

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    count: Mapped[int] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())


class ModelDB(Base):
    __tablename__ = "models"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    architecture: Mapped[str] = mapped_column(String(64), nullable=False)
    format: Mapped[str] = mapped_column(String(64), nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    export_path: Mapped[str] = mapped_column(Text, nullable=False)
    is_ready: Mapped[bool] = mapped_column(Boolean, nullable=False)
    size: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    backbone: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    train_job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="RESTRICT"))
    dataset_snapshot_id: Mapped[str] = mapped_column(ForeignKey("dataset_snapshot.id", ondelete="RESTRICT"))

    dataset_snapshot = relationship("DatasetSnapshotDB", uselist=False, lazy="joined")


class JobDB(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    progress: Mapped[int] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    start_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    end_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    payload: Mapped[str] = mapped_column(JSON, nullable=False)


class PipelineDB(Base):
    __tablename__ = "pipelines"

    project_id: Mapped[str] = mapped_column(Text, ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True)
    source_id: Mapped[str | None] = mapped_column(Text, ForeignKey("sources.id", ondelete="RESTRICT"))
    sink_id: Mapped[str | None] = mapped_column(Text, ForeignKey("sinks.id", ondelete="RESTRICT"))
    model_id: Mapped[str | None] = mapped_column(Text, ForeignKey("models.id", ondelete="RESTRICT"))
    inference_device: Mapped[str] = mapped_column(String(64), nullable=False, default="CPU")
    overlay: Mapped[bool | None] = mapped_column(Boolean, default=True)
    is_running: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    data_collection_policies: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())

    project = relationship("ProjectDB", back_populates="pipeline", lazy="joined")
    sink = relationship("SinkDB", uselist=False, lazy="joined")
    source = relationship("SourceDB", uselist=False, lazy="joined")
    model = relationship("ModelDB", uselist=False, lazy="joined")


class SourceDB(Base):
    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    config_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())


class SinkDB(Base):
    __tablename__ = "sinks"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    sink_type: Mapped[str] = mapped_column(String(50), nullable=False)
    rate_limit: Mapped[float | None] = mapped_column(Float, nullable=True)
    config_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    output_formats: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())
