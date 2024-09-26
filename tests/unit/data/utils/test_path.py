"""Tests for path utils."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from anomalib.data.utils.path import validate_path


class TestValidatePath:
    """Tests for ``validate_path`` function."""

    @staticmethod
    def test_invalid_path_type() -> None:
        """Test ``validate_path`` raises TypeError for an invalid path type."""
        with pytest.raises(TypeError, match=r"Expected str, bytes or os.PathLike object, not*"):
            validate_path(123)

    @staticmethod
    def test_is_path_too_long() -> None:
        """Test ``validate_path`` raises ValueError for a path that is too long."""
        with pytest.raises(ValueError, match=r"Path is too long: *"):
            validate_path("/" * 1000)

    @staticmethod
    def test_contains_non_printable_characters() -> None:
        """Test ``validate_path`` raises ValueError for a path that contains non-printable characters."""
        with pytest.raises(ValueError, match=r"Path contains non-printable characters: *"):
            validate_path("/\x00")

    @staticmethod
    def test_existing_file_within_base_dir(dataset_path: Path) -> None:
        """Test ``validate_path`` returns the validated path for an existing file within the base directory."""
        file_path = dataset_path / "mvtec/dummy/train/good/000.png"
        validated_path = validate_path(file_path, base_dir=dataset_path)
        assert validated_path == file_path.resolve()

    @staticmethod
    def test_existing_directory_within_base_dir(dataset_path: Path) -> None:
        """Test ``validate_path`` returns the validated path for an existing directory within the base directory."""
        directory_path = dataset_path / "mvtec/dummy/train/good"
        validated_path = validate_path(directory_path, base_dir=dataset_path)
        assert validated_path == directory_path.resolve()

    @staticmethod
    def test_nonexistent_file(dataset_path: Path) -> None:
        """Test ``validate_path`` raises FileNotFoundError for a nonexistent file."""
        with pytest.raises(FileNotFoundError):
            validate_path(dataset_path / "nonexistent/file.png")

    @staticmethod
    def test_nonexistent_directory(dataset_path: Path) -> None:
        """Test ``validate_path`` raises FileNotFoundError for a nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            validate_path(dataset_path / "nonexistent/directory")

    @staticmethod
    def test_no_read_permission() -> None:
        """Test ``validate_path`` raises PermissionError for a file without read permission."""
        with TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test.txt"
            with file_path.open("w") as f:
                f.write("test")
            file_path.chmod(0o222)  # Remove read permission
            with pytest.raises(PermissionError, match=r"Read or execute permissions denied for the path:*"):
                validate_path(file_path, base_dir=Path(tmp_dir))

    @staticmethod
    def test_no_read_execute_permission() -> None:
        """Test ``validate_path`` raises PermissionError for a directory without read and execute permission."""
        with TemporaryDirectory() as tmp_dir:
            Path(tmp_dir).chmod(0o222)  # Remove read and execute permission
            with pytest.raises(PermissionError, match=r"Read or execute permissions denied for the path:*"):
                validate_path(tmp_dir, base_dir=Path(tmp_dir))

    @staticmethod
    def test_file_wrongsuffix() -> None:
        """Test ``validate_path`` raises ValueError for a file with wrong suffix."""
        with pytest.raises(ValueError, match="Path extension is not accepted."):
            validate_path("file.png", should_exist=False, extensions=(".json", ".txt"))
