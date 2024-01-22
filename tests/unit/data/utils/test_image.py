"""Tests for image utils."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from anomalib.data.utils.image import get_image_filenames
from anomalib.data.utils.path import validate_path


class TestValidatePath:
    """Tests for ``validate_path`` function."""

    def test_invalid_path_type(self) -> None:
        """Test ``validate_path`` raises TypeError for an invalid path type."""
        with pytest.raises(TypeError, match=r"Expected str, bytes or os.PathLike object, not*"):
            validate_path(123)

    def test_existing_file_within_base_dir(self, dataset_path: Path) -> None:
        """Test ``validate_path`` returns the validated path for an existing file within the base directory."""
        file_path = dataset_path / "mvtec/dummy/train/good/000.png"
        validated_path = validate_path(file_path, base_dir=dataset_path)
        assert validated_path == file_path.resolve()

    def test_existing_directory_within_base_dir(self, dataset_path: Path) -> None:
        """Test ``validate_path`` returns the validated path for an existing directory within the base directory."""
        directory_path = dataset_path / "mvtec/dummy/train/good"
        validated_path = validate_path(directory_path, base_dir=dataset_path)
        assert validated_path == directory_path.resolve()

    def test_nonexistent_file(self, dataset_path: Path) -> None:
        """Test ``validate_path`` raises FileNotFoundError for a nonexistent file."""
        with pytest.raises(FileNotFoundError):
            validate_path(dataset_path / "nonexistent/file.png")

    def test_nonexistent_directory(self, dataset_path: Path) -> None:
        """Test ``validate_path`` raises FileNotFoundError for a nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            validate_path(dataset_path / "nonexistent/directory")

    def test_outside_base_dir(self) -> None:
        """Test ``validate_path`` raises ValueError for a path outside the base directory."""
        with pytest.raises(ValueError, match=r"Access denied: Path is outside the allowed directory"):
            validate_path("/usr/local/lib")

    def test_no_read_permission(self) -> None:
        """Test ``validate_path`` raises PermissionError for a file without read permission."""
        with TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test.txt"
            with file_path.open("w") as f:
                f.write("test")
            file_path.chmod(0o222)  # Remove read permission
            with pytest.raises(PermissionError, match=r"Read or execute permissions denied for the path:*"):
                validate_path(file_path, base_dir=Path(tmp_dir))

    def test_no_read_execute_permission(self) -> None:
        """Test ``validate_path`` raises PermissionError for a directory without read and execute permission."""
        with TemporaryDirectory() as tmp_dir:
            Path(tmp_dir).chmod(0o222)  # Remove read and execute permission
            with pytest.raises(PermissionError, match=r"Read or execute permissions denied for the path:*"):
                validate_path(tmp_dir, base_dir=Path(tmp_dir))


class TestGetImageFilenames:
    """Tests for ``get_image_filenames`` function."""

    def test_existing_image_file(self, dataset_path: Path) -> None:
        """Test ``get_image_filenames`` returns the correct path for an existing image file."""
        image_path = dataset_path / "mvtec/dummy/train/good/000.png"
        image_filenames = get_image_filenames(image_path)
        assert image_filenames == [image_path.resolve()]

    def test_existing_image_directory(self, dataset_path: Path) -> None:
        """Test ``get_image_filenames`` returns the correct image filenames from an existing directory."""
        directory_path = dataset_path / "mvtec/dummy/train/good"
        image_filenames = get_image_filenames(directory_path)
        expected_filenames = [(directory_path / f"{i:03d}.png").resolve() for i in range(5)]
        assert set(image_filenames) == set(expected_filenames)

    def test_nonexistent_image_file(self) -> None:
        """Test ``get_image_filenames`` raises FileNotFoundError for a nonexistent image file."""
        with pytest.raises(FileNotFoundError):
            get_image_filenames("009.tiff")

    def test_nonexistent_image_directory(self) -> None:
        """Test ``get_image_filenames`` raises FileNotFoundError for a nonexistent image directory."""
        with pytest.raises(FileNotFoundError):
            get_image_filenames("nonexistent_directory")

    def test_non_image_file(self, dataset_path: Path) -> None:
        """Test ``get_image_filenames`` raises ValueError for a non-image file."""
        filename = dataset_path / "avenue/ground_truth_demo/testing_label_mask/1_label.mat"
        with pytest.raises(ValueError, match=r"``filename`` is not an image file*"):
            get_image_filenames(filename)

    def test_outside_base_dir(self) -> None:
        """Test ``get_image_filenames`` raises ValueError for a path outside the base directory."""
        with TemporaryDirectory() as tmp_dir, pytest.raises(
            ValueError,
            match=r"Access denied: Path is outside the allowed directory",
        ):
            get_image_filenames(tmp_dir, base_dir=Path.home())
