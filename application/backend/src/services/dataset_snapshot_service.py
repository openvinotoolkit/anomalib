# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
import io
import os
import tempfile
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio.session import AsyncSession

from db import get_async_db_session_ctx
from db.schema import ModelDB
from pydantic_models import DatasetSnapshot
from pydantic_models.base import Pagination
from pydantic_models.dataset_snapshot import DatasetSnapshotList
from repositories import DatasetSnapshotRepository, MediaRepository, ProjectRepository
from repositories.binary_repo import DatasetSnapshotBinaryRepository, ImageBinaryRepository


class DatasetSnapshotService:
    @classmethod
    async def create_snapshot(cls, project_id: UUID) -> DatasetSnapshot:
        """Create a new immutable dataset snapshot from current media."""
        snapshot_id = uuid4()
        filename = f"{snapshot_id}.parquet"

        logger.info(f"Creating dataset snapshot {snapshot_id} for project {project_id}")

        async with get_async_db_session_ctx() as session:
            media_repo = MediaRepository(session, project_id=project_id)
            image_bin_repo = ImageBinaryRepository(project_id=project_id)
            data_rows = []
            async for item in media_repo.get_all_streaming():
                # Read bytes
                try:
                    img_bytes = await image_bin_repo.read_file(item.filename)
                except FileNotFoundError:
                    logger.error(f"Image file {item.filename} missing for media item `{item.id}`, skipping")
                    continue

                data_rows.append({
                    "image": img_bytes,
                    "is_anomalous": item.is_anomalous,
                    "filename": item.filename,  # Virtual path, useful for debugging
                    # "mask": ... # TODO: Add mask support if we have masks
                })

        # Create Table
        if not data_rows:
            logger.warning(f"Creating snapshot for project {project_id} with no media")
            # Create empty table with schema
            schema = pa.schema([
                ("image", pa.binary()),
                ("is_anomalous", pa.bool_()),
                ("filename", pa.string()),
            ])
            table = pa.Table.from_pydict({"image": [], "is_anomalous": [], "filename": []}, schema=schema)
        else:
            # Convert to PyArrow Table
            pydict = {
                "image": [r["image"] for r in data_rows],
                "is_anomalous": [r["is_anomalous"] for r in data_rows],
                "filename": [r["filename"] for r in data_rows],
            }
            table = pa.Table.from_pydict(pydict)

        # Write Parquet file
        snapshot_bin_repo = DatasetSnapshotBinaryRepository(project_id=project_id)

        def _write_parquet() -> bytes:
            sink = pa.BufferOutputStream()
            pq.write_table(table, sink)
            return sink.getvalue().to_pybytes()

        parquet_bytes = await asyncio.to_thread(_write_parquet)

        try:
            await snapshot_bin_repo.save_file(filename, parquet_bytes)
            logger.info(f"Saved snapshot file {filename}")

            # Create snapshot DB Record
            async with get_async_db_session_ctx() as session:
                snapshot_repo = DatasetSnapshotRepository(session, project_id=project_id)
                snapshot = DatasetSnapshot(
                    id=snapshot_id,
                    project_id=project_id,
                    filename=filename,
                    count=len(data_rows),
                )
                return await snapshot_repo.save(snapshot)

        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            # Cleanup file if saved
            try:
                await snapshot_bin_repo.delete_file(filename)
                logger.info(f"Deleted incomplete snapshot file {filename}")
            except FileNotFoundError:
                logger.info(f"No incomplete snapshot file {filename} to delete")
            raise e

    @classmethod
    async def get_or_create_snapshot(cls, project_id: UUID, snapshot_id: UUID | None) -> DatasetSnapshot:
        """Get existing snapshot or create a new one:
        - If snapshot_id is provided, fetch and return it.
        - If snapshot_id is None, create a new snapshot if necessary.
        - If dataset has not being updated since last snapshot, return existing snapshot.
        """
        async with get_async_db_session_ctx() as session:
            snapshot_repo = DatasetSnapshotRepository(session, project_id=project_id)
            if snapshot_id is not None:
                snapshot = await snapshot_repo.get_by_id(snapshot_id)
                if snapshot is None:
                    raise ValueError(f"Snapshot {snapshot_id} not found in project {project_id}")
                return snapshot

            dataset_updated_at = await ProjectRepository(session).get_dataset_timestamp(project_id=project_id)
            latest_snapshot = await snapshot_repo.get_latest_snapshot()
            if latest_snapshot and latest_snapshot.created_at and latest_snapshot.created_at >= dataset_updated_at:
                logger.info(f"Using existing up-to-date snapshot `{latest_snapshot.id}` in project `{project_id}`")
                return latest_snapshot
        # Create new snapshot only if no up-to-date snapshot exists
        logger.info(f"Creating new snapshot for project `{project_id}`")
        return await cls.create_snapshot(project_id=project_id)

    @classmethod
    async def delete_snapshot_if_unused(cls, snapshot_id: UUID, project_id: UUID) -> None:
        """Delete snapshot only if no models reference it."""
        async with get_async_db_session_ctx() as session:
            # Check references
            stmt = select(func.count(ModelDB.id)).where(ModelDB.dataset_snapshot_id == str(snapshot_id))
            count = (await session.execute(stmt)).scalar() or 0

            if count > 0:
                logger.info(f"Snapshot {snapshot_id} is used by {count} models. Skipping delete.")
                return

            logger.info(f"Snapshot {snapshot_id} is unused. Deleting.")

            # Delete DB Record
            snapshot_repo = DatasetSnapshotRepository(session, project_id=project_id)
            snapshot = await snapshot_repo.get_by_id(snapshot_id)

            if snapshot:
                await snapshot_repo.delete_by_id(snapshot_id)

        # Delete file (outside DB session)
        if snapshot:
            snapshot_bin_repo = DatasetSnapshotBinaryRepository(project_id=project_id)
            try:
                await snapshot_bin_repo.delete_file(snapshot.filename)
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.error(f"Error deleting snapshot file {snapshot.filename}: {e}")

    @classmethod
    async def delete_project_snapshots_db(cls, session: AsyncSession, project_id: UUID, commit: bool = False) -> None:
        """Delete all snapshots associated with a project from the database."""
        snapshot_repo = DatasetSnapshotRepository(session, project_id=project_id)
        await snapshot_repo.delete_all(commit=commit)

    @classmethod
    async def cleanup_project_snapshot_files(cls, project_id: UUID) -> None:
        """Cleanup snapshot files for a project."""
        try:
            # Cleanup project folder (removes all files at once)
            snapshot_bin_repo = DatasetSnapshotBinaryRepository(project_id=project_id)
            await snapshot_bin_repo.delete_project_folder()
            logger.info(f"Cleaned up snapshot files for project {project_id}")
        except Exception as e:
            logger.warning(f"Failed to cleanup snapshot files for project {project_id}: {e}")

    @staticmethod
    def extract_snapshot_to_path(snapshot_path: str, temp_dir: str) -> None:
        """Extract images from Parquet snapshot to a temporary directory structure."""
        logger.info(f"Extracting snapshot {snapshot_path} to {temp_dir}")

        # Ensure directories exist
        normal_dir = os.path.join(temp_dir, "normal")
        abnormal_dir = os.path.join(temp_dir, "abnormal")
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(abnormal_dir, exist_ok=True)

        # Read parquet in batches
        parquet_file = pq.ParquetFile(snapshot_path)

        # Iterate over batches to avoid OOM with large datasets
        for batch in parquet_file.iter_batches(batch_size=100):
            df = batch.to_pandas()

            for row in df.itertuples(index=False):
                try:
                    is_anomalous = row.is_anomalous
                    filename = os.path.basename(row.filename)
                    target_dir = abnormal_dir if is_anomalous else normal_dir
                    save_path = os.path.join(target_dir, filename)

                    # Write image
                    image_data = row.image
                    if isinstance(image_data, (bytes, bytearray)):
                        image = Image.open(io.BytesIO(image_data))
                        image.save(save_path)
                    else:
                        logger.error(f"Image type {type(image_data)} is not supported.")
                        raise ValueError(f"Image type {type(image_data)} is not supported.")
                except AttributeError as e:
                    logger.error(f"Snapshot file `{snapshot_path}` has missing expected columns: {e}")
                    raise e

    @classmethod
    @asynccontextmanager
    async def use_snapshot_as_folder(cls, snapshot_id: UUID, project_id: UUID):
        """Context manager to use a snapshot as a temporary folder."""
        snapshot_bin_repo = DatasetSnapshotBinaryRepository(project_id=project_id)
        snapshot_path = snapshot_bin_repo.get_snapshot_path(snapshot_id)

        with tempfile.TemporaryDirectory() as temp_dir:
            await asyncio.to_thread(cls.extract_snapshot_to_path, snapshot_path, temp_dir)
            yield temp_dir

    @staticmethod
    async def list_snapshots(project_id: UUID, limit: int, offset: int) -> DatasetSnapshotList:
        """List all dataset snapshots for a project."""
        async with get_async_db_session_ctx() as session:
            repo = DatasetSnapshotRepository(session, project_id=project_id)
            total = await repo.get_all_count()
            items = await repo.get_all_pagination(limit=limit, offset=offset)
        return DatasetSnapshotList(
            snapshots=items,
            pagination=Pagination(
                limit=limit,
                offset=offset,
                count=len(items),
                total=total,
            ),
        )

    @staticmethod
    async def get_snapshot(project_id: UUID, snapshot_id: UUID) -> DatasetSnapshot:
        """Get dataset snapshot by ID."""
        async with get_async_db_session_ctx() as session:
            snapshot_repo = DatasetSnapshotRepository(session, project_id=project_id)
            snapshot = await snapshot_repo.get_by_id(snapshot_id)
            if snapshot is None:
                raise ValueError(f"Snapshot {snapshot_id} not found in project {project_id}")
            return snapshot
