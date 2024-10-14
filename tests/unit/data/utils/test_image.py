"""Tests for image utils."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from anomalib.data.utils.image import get_image_filenames


class TestGetImageFilenames:
    """Tests for ``get_image_filenames`` function."""

    @staticmethod
    def test_existing_image_file(dataset_path: Path) -> None:
        """Test ``get_image_filenames`` returns the correct path for an existing image file."""
        image_path = dataset_path / "mvtec/dummy/train/good/000.png"
        image_filenames = get_image_filenames(image_path)
        assert image_filenames == [image_path.resolve()]

    @staticmethod
    def test_existing_image_directory(dataset_path: Path) -> None:
        """Test ``get_image_filenames`` returns the correct image filenames from an existing directory."""
        directory_path = dataset_path / "mvtec/dummy/train/good"
        image_filenames = get_image_filenames(directory_path)
        expected_filenames = [(directory_path / f"{i:03d}.png").resolve() for i in range(5)]
        assert set(image_filenames) == set(expected_filenames)

    @staticmethod
    def test_nonexistent_image_file() -> None:
        """Test ``get_image_filenames`` raises FileNotFoundError for a nonexistent image file."""
        with pytest.raises(FileNotFoundError):
            get_image_filenames("009.tiff")

    @staticmethod
    def test_nonexistent_image_directory() -> None:
        """Test ``get_image_filenames`` raises FileNotFoundError for a nonexistent image directory."""
        with pytest.raises(FileNotFoundError):
            get_image_filenames("nonexistent_directory")

    @staticmethod
    def test_non_image_file(dataset_path: Path) -> None:
        """Test ``get_image_filenames`` raises ValueError for a non-image file."""
        filename = dataset_path / "avenue/ground_truth_demo/testing_label_mask/1_label.mat"
        with pytest.raises(ValueError, match=r"``filename`` is not an image file*"):
            get_image_filenames(filename)
