# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from uuid import UUID

from db.schema import DatasetSnapshotDB
from pydantic_models.dataset_snapshot import DatasetSnapshot
from repositories.mappers.base_mapper_interface import IBaseMapper


class DatasetSnapshotMapper(IBaseMapper[DatasetSnapshotDB, DatasetSnapshot]):
    """Mapper for DatasetSnapshot entity <-> DB schema conversions."""

    @staticmethod
    def to_schema(model: DatasetSnapshot) -> DatasetSnapshotDB:
        return DatasetSnapshotDB(
            id=str(model.id),
            project_id=str(model.project_id),
            filename=model.filename,
            count=model.count,
            created_at=model.created_at,
        )

    @staticmethod
    def from_schema(schema: DatasetSnapshotDB) -> DatasetSnapshot:
        return DatasetSnapshot(
            id=UUID(schema.id),
            project_id=UUID(schema.project_id),
            filename=schema.filename,
            count=schema.count,
            created_at=schema.created_at,
        )
