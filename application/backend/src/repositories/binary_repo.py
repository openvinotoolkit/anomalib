# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
import asyncio
import os
import shutil
from enum import StrEnum
from functools import cached_property
from typing import TYPE_CHECKING

from settings import get_settings

if TYPE_CHECKING:
    from uuid import UUID

    from pydantic_models.model import ExportParameters
    from settings import Settings

settings: Settings = get_settings()


class FileType(StrEnum):
    IMAGES = "images"
    MODELS = "models"
    SNAPSHOTS = "snapshots"
    MODEL_EXPORTS = "model_exports"
    VIDEOS = "videos"


class BinaryRepository(metaclass=abc.ABCMeta):
    def __init__(self, project_id: str | UUID, file_type: FileType):
        self.project_id = str(project_id)
        self.file_type = file_type

    def _assert_within_project(self, full_path: str) -> None:
        """
        Assert that a resolved path is contained within the project folder.

        Args:
            full_path (str): Absolute, already-resolved filesystem path to check.

        Raises:
            ValueError: If the path escapes the project directory.
        """
        project_root = os.path.realpath(self.project_folder_path)
        try:
            common = os.path.commonpath([project_root, full_path])
        except ValueError as exc:
            raise ValueError("Invalid filename: path traversal detected") from exc
        if common != project_root:
            raise ValueError("Invalid filename: path traversal detected")

    async def read_file(self, filename: str) -> bytes:
        """
        Read a binary file from the filesystem.

        Args:
            filename: Relative path to the file.

        Returns:
            Binary content of the file.
        """

        def stdlib_read():
            full_path = os.path.realpath(self.get_full_path(filename))
            self._assert_within_project(full_path)
            if not os.path.isfile(full_path):
                raise FileNotFoundError(f"File not found: {full_path}")
            with open(full_path, "rb") as fp:
                return fp.read()

        return await asyncio.to_thread(stdlib_read)

    @cached_property
    def project_folder_path(self) -> str:
        """
        Get the project folder path containing the binary files.
        """
        return os.path.join(settings.data_dir, self.file_type, "projects", self.project_id)

    @abc.abstractmethod
    def get_full_path(self, filename: str) -> str:
        """
        Get the full path for a given filename within the project folder.

        Args:
            filename: Name of the file.

        Returns:
            Full path to the file.
        """

    async def save_file(self, filename: str, content: bytes) -> str:
        """
        Save a binary file to the filesystem under the project directory.

        Args:
            filename: Name of the file to save.
            content: Binary content of the file.

        Returns:
            The path where the file was saved.
        """

        def stdlib_write():
            full_path = os.path.realpath(self.get_full_path(filename))
            self._assert_within_project(full_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "wb") as f:
                f.write(content)
            return full_path

        try:
            destination_path = await asyncio.to_thread(stdlib_write)
        except Exception as e:
            raise OSError(f"Failed to save {self.file_type} file: {filename}") from e
        return destination_path

    async def delete_file(self, filename: str) -> None:
        """
        Delete a binary file from the filesystem.

        Args:
            filename: Name of the file to delete.
        """

        def stdlib_delete():
            full_path = os.path.realpath(self.get_full_path(filename))
            self._assert_within_project(full_path)
            if os.path.isfile(full_path):
                os.remove(full_path)
            else:
                raise FileNotFoundError(f"File not found: {full_path}")

        await asyncio.to_thread(stdlib_delete)

    async def delete_project_folder(self) -> None:
        """
        Delete the entire project folder for this file type.
        """

        def stdlib_delete_folder():
            if os.path.exists(self.project_folder_path):
                shutil.rmtree(self.project_folder_path)

        await asyncio.to_thread(stdlib_delete_folder)


class DatasetSnapshotBinaryRepository(BinaryRepository):
    def __init__(self, project_id: str | UUID):
        super().__init__(project_id=project_id, file_type=FileType.SNAPSHOTS)

    def get_full_path(self, filename: str) -> str:
        return os.path.join(self.project_folder_path, filename)

    def get_snapshot_path(self, snapshot_id: str | UUID) -> str:
        """
        Get the full path for a dataset snapshot.

        Args:
            snapshot_id: ID of the snapshot.

        Returns:
            Full path to the snapshot file.
        """
        return self.get_full_path(f"{snapshot_id}.parquet")


class ImageBinaryRepository(BinaryRepository):
    def __init__(self, project_id: str | UUID):
        super().__init__(project_id=project_id, file_type=FileType.IMAGES)

    def get_full_path(self, filename: str) -> str:
        return os.path.join(self.project_folder_path, filename)


class VideoBinaryRepository(BinaryRepository):
    """Binary repository for storing and retrieving video files for a project.

    This repository manages video binaries under the project-specific ``videos``
    directory. Use :meth:`get_full_path` to resolve a video filename to its
    absolute path on the filesystem.
    """

    def __init__(self, project_id: str | UUID):
        super().__init__(project_id=project_id, file_type=FileType.VIDEOS)

    def get_full_path(self, filename: str) -> str:
        return os.path.join(self.project_folder_path, filename)


class ModelBinaryRepository(BinaryRepository):
    def __init__(self, project_id: str | UUID, model_id: str | UUID):
        super().__init__(project_id=project_id, file_type=FileType.MODELS)
        self._model_id = str(model_id)

    def get_full_path(self, filename: str) -> str:
        return os.path.join(self.model_folder_path, filename)

    @cached_property
    def model_folder_path(self) -> str:
        """
        Get the folder path for models.

        Returns:
            Folder path for models.
        """
        return os.path.join(self.project_folder_path, self._model_id)

    async def delete_model_folder(self) -> None:
        """
        Delete a model folder from the filesystem.
        """

        def stdlib_delete_folder():
            folder_path = self.model_folder_path
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            else:
                raise FileNotFoundError(f"Model folder not found: {folder_path}")

        await asyncio.to_thread(stdlib_delete_folder)

    def get_weights_file_path(self, name: str) -> str:
        """
        Read a weights file from the model folder.

        Args:
            name: Name of the weights to read.

        Returns:
            path of the weights file.
        """
        return os.path.join(self.model_folder_path, name)


class ModelExportBinaryRepository(BinaryRepository):
    def __init__(self, project_id: str | UUID, model_id: str | UUID):
        super().__init__(project_id=project_id, file_type=FileType.MODEL_EXPORTS)
        self._model_id = str(model_id)

    def get_full_path(self, filename: str) -> str:
        return os.path.join(self.model_export_folder_path, filename)

    @cached_property
    def model_export_folder_path(self) -> str:
        """
        Get the folder path for model exports.

        Returns:
            Folder path for model exports.
        """
        return os.path.join(self.project_folder_path, self._model_id)

    def get_model_export_path(self, model_name: str, export_params: ExportParameters) -> str:
        """
        Get the full path for a dataset snapshot.

        Args:
            model_name: name of the model
            export_params: model export parameters

        Returns:
            Full path to the model export zip file.
        """
        compression_suffix = f"_{export_params.compression.value}" if export_params.compression else ""
        filename = f"{model_name}_{export_params.format.value}{compression_suffix}.zip"
        return self.get_full_path(filename)
