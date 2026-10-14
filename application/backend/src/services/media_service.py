# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
from io import BytesIO
from uuid import UUID, uuid4

import numpy as np
from loguru import logger
from PIL import Image
from sqlalchemy.ext.asyncio.session import AsyncSession

from db import get_async_db_session_ctx
from pydantic_models import Media, MediaList
from pydantic_models.base import Pagination
from repositories import MediaRepository, ProjectRepository
from repositories.binary_repo import ImageBinaryRepository
from services import ResourceNotFoundError
from services.exceptions import ResourceType

THUMBNAIL_SIZE = 256  # Max width/height for thumbnails in pixels


class MediaService:
    @staticmethod
    def _get_thumbnail_filename(media_id: UUID) -> str:
        """Generate thumbnail filename for a given media ID."""
        return f"thumb_{media_id}.png"

    @staticmethod
    async def get_media_list(project_id: UUID, limit: int, offset: int) -> MediaList:
        async with get_async_db_session_ctx() as session:
            repo = MediaRepository(session, project_id=project_id)
            total = await repo.get_all_count()
            items = await repo.get_all_pagination(limit=limit, offset=offset)
        return MediaList(
            media=items,
            pagination=Pagination(
                limit=limit,
                offset=offset,
                count=len(items),
                total=total,
            ),
        )

    @staticmethod
    async def get_media_by_id(project_id: UUID, media_id: UUID) -> Media | None:
        async with get_async_db_session_ctx() as session:
            repo = MediaRepository(session, project_id=project_id)
            return await repo.get_by_id(media_id)

    @classmethod
    async def get_media_file_path(cls, project_id: UUID, media_id: UUID) -> str:
        media = await cls.get_media_by_id(project_id=project_id, media_id=media_id)
        if media is None:
            raise FileNotFoundError(f"Media with ID {media_id} not found.")
        bin_repo = ImageBinaryRepository(project_id=project_id)
        return bin_repo.get_full_path(filename=media.filename)

    @classmethod
    async def get_thumbnail_file_path(cls, project_id: UUID, media_id: UUID) -> str:
        """Get the file path for a media's thumbnail."""
        media = await cls.get_media_by_id(project_id=project_id, media_id=media_id)
        if media is None:
            raise FileNotFoundError(f"Media with ID {media_id} not found.")
        bin_repo = ImageBinaryRepository(project_id=project_id)
        thumbnail_filename = cls._get_thumbnail_filename(media_id)
        return bin_repo.get_full_path(filename=thumbnail_filename)

    @classmethod
    async def upload_image(
        cls,
        project_id: UUID,
        image: np.ndarray | bytes,
        is_anomalous: bool,
        extension: str | None = None,
        size: int | None = None,
    ) -> Media:
        # Generate unique filename and media ID
        media_id = uuid4()

        if not extension or not extension.lstrip("."):
            raise ValueError("File extension must be provided")

        if isinstance(image, np.ndarray):

            def _encode_image() -> bytes:
                with BytesIO() as output:
                    # Determine format from extension, default to PNG if unknown or generic
                    fmt = extension.lstrip(".").upper()
                    Image.fromarray(image).save(output, format=fmt)
                    return output.getvalue()

            image_bytes = await asyncio.to_thread(_encode_image)
        else:
            image_bytes = image

        filename = f"{media_id}{extension}"
        bin_repo = ImageBinaryRepository(project_id=project_id)
        saved_media: Media | None = None

        # Extract image dimensions
        def _get_image_size() -> tuple[int, int]:
            with Image.open(BytesIO(image_bytes)) as img:
                return img.size  # Returns (width, height)

        width, height = await asyncio.to_thread(_get_image_size)

        try:
            # Save original file to filesystem
            saved_file_path = await bin_repo.save_file(
                filename=filename,
                content=image_bytes,
            )
            logger.info(f"Saved media file: {saved_file_path}")

            # Create media record in database
            media = Media(
                id=media_id,
                project_id=project_id,
                filename=filename,
                size=size or len(image_bytes),
                is_anomalous=is_anomalous,
                width=width,
                height=height,
            )
            async with get_async_db_session_ctx() as session:
                media_repo = MediaRepository(session, project_id=project_id)
                saved_media = await media_repo.save(media)
                await ProjectRepository(session).update_dataset_timestamp(project_id=project_id)
        except Exception as e:
            logger.error(f"Rolling back media upload due to error: {e}")
            # Attempt to delete the files if they were saved
            try:
                await cls._delete_media_file(project_id=project_id, filename=filename)
            except Exception as delete_error:
                logger.error(f"Failed to delete media file during rollback: {delete_error}")
            if saved_media is not None:
                async with get_async_db_session_ctx() as session:
                    await MediaRepository(session, project_id=project_id).delete_by_id(saved_media.id)
            raise e
        return saved_media

    @classmethod
    async def delete_media(cls, media_id: UUID, project_id: UUID) -> None:
        async with get_async_db_session_ctx() as session:
            media_repo = MediaRepository(session, project_id=project_id)
            media = await media_repo.get_by_id(media_id)
            if media is None:
                raise ResourceNotFoundError(resource_type=ResourceType.MEDIA, resource_id=str(media_id))
            await media_repo.delete_by_id(media_id)
        thumbnail_filename = cls._get_thumbnail_filename(media_id)
        await cls._delete_media_file(project_id=project_id, filename=media.filename)
        await cls._delete_media_file(project_id=project_id, filename=thumbnail_filename)
        await ProjectRepository(session).update_dataset_timestamp(project_id=project_id)

    @classmethod
    async def delete_project_media_db(cls, session: AsyncSession, project_id: UUID, commit: bool = False) -> None:
        """Delete all media associated with a project from the database."""
        media_repo = MediaRepository(session, project_id=project_id)
        await media_repo.delete_all(commit=commit)

    @classmethod
    async def cleanup_project_media_files(cls, project_id: UUID) -> None:
        """Cleanup media files for a project."""
        try:
            # Cleanup project folder (removes all files at once)
            bin_repo = ImageBinaryRepository(project_id=project_id)
            await bin_repo.delete_project_folder()
            logger.info(f"Cleaned up media files for project {project_id}")
        except Exception as e:
            logger.warning(f"Failed to cleanup media files for project {project_id}: {e}")

    @staticmethod
    async def _delete_media_file(project_id: UUID, filename: str) -> None:
        bin_repo = ImageBinaryRepository(project_id=project_id)

        try:
            await bin_repo.delete_file(filename=filename)
        except FileNotFoundError:
            logger.warning(f"File `{filename}` not found during media deletion.")
        except Exception as e:
            logger.error(f"Error deleting file `{filename}`: {e}")
            raise e

    @classmethod
    async def generate_thumbnail(
        cls,
        project_id: UUID,
        media_id: UUID,
        image: np.ndarray | bytes,
        height_px: int = THUMBNAIL_SIZE,
        width_px: int = THUMBNAIL_SIZE,
    ) -> None:
        """Create and persist a PNG thumbnail for a media item.

        Resizes the input image to fit within width_px x height_px while
        preserving aspect ratio. Intended to run as a background task so the
        upload path stays fast.

        Args:
            project_id: Identifier of the owning project.
            media_id: Identifier of the media item the thumbnail belongs to.
            image: Original image (bytes or numpy array) used to create the thumbnail.
            height_px: Maximum thumbnail height in pixels. Defaults to
                ``THUMBNAIL_SIZE``.
            width_px: Maximum thumbnail width in pixels. Defaults to
                ``THUMBNAIL_SIZE``.

        Returns:
            None

        Raises:
            Exception: If processing or persistence fails
        """

        def _create() -> bytes:
            if isinstance(image, np.ndarray):
                img = Image.fromarray(image)
            else:
                img = Image.open(BytesIO(image))

            with img:
                # Preserve aspect ratio while fitting within the box
                img.thumbnail((width_px, height_px))
                with BytesIO() as buf:
                    img.save(buf, format="PNG")
                    return buf.getvalue()

        thumbnail_bytes = await asyncio.to_thread(_create)
        thumbnail_filename = cls._get_thumbnail_filename(media_id)
        bin_repo = ImageBinaryRepository(project_id=project_id)
        try:
            thumbnail_path = await bin_repo.save_file(
                filename=thumbnail_filename,
                content=thumbnail_bytes,
            )
            logger.info(f"Saved thumbnail: {thumbnail_path}")
        except Exception as e:
            logger.error(f"Rolling back media thumbnail generation due to error: {e}")
            # Attempt to delete the files if they were saved
            try:
                await cls._delete_media_file(project_id=project_id, filename=thumbnail_filename)
            except Exception as delete_error:
                logger.error(f"Failed to delete media file during rollback: {delete_error}")
            raise e
