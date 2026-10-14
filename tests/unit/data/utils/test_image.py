# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for image utils."""

from collections.abc import Callable
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from anomalib.data.utils.image import Image, get_image_filenames, np, read_mask, torch


class TestGetImageFilenames:
    """Tests for ``get_image_filenames`` function."""

    @staticmethod
    def test_existing_image_file(mvtecad_path: Path) -> None:
        """Test ``get_image_filenames`` returns the correct path for an existing image file."""
        image_path = mvtecad_path / "dummy/train/good/000.png"
        image_filenames = get_image_filenames(image_path)
        assert image_filenames == [image_path.resolve()]

    @staticmethod
    def test_existing_image_directory(mvtecad_path: Path) -> None:
        """Test ``get_image_filenames`` returns the correct image filenames from an existing directory."""
        directory_path = mvtecad_path / "dummy/train/good"
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
    def test_non_image_file(dataset_path: Path, make_dummy_dataset: Callable[[str], Path]) -> None:
        """Test ``get_image_filenames`` raises ValueError for a non-image file."""
        make_dummy_dataset("avenue")
        filename = dataset_path / "avenue/ground_truth_demo/testing_label_mask/1_label.mat"
        with pytest.raises(ValueError, match=r"``filename`` is not an image file*"):
            get_image_filenames(filename)


class TestReadMask:
    """Tests for ``read_mask`` function."""

    @staticmethod
    def test_mask_conversion_as_numpy(mocker: MockerFixture) -> None:
        """Test that the function correctly converts the image to a numpy array."""
        # Sample image: [[0, 236], [255, 0]]
        mocker.patch(
            "anomalib.data.utils.image.Image.open",
            return_value=Image.fromarray(np.array([[0, 236], [255, 0]], dtype=np.uint8)),
        )

        # Mocks are setup, call function to test!
        result = read_mask("dummy_path.png", as_tensor=False)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([[0, 236], [255, 0]]))

    @staticmethod
    def test_mask_conversion_as_tensor(mocker: MockerFixture) -> None:
        """Test that the function correctly converts the image to a tensor (normalized to 0, 1).

        Test cases:
        - Value 0 remains non-anomalous (0)
        - Value 236 (>0) becomes anomalous (1)
        - Value 255 becomes anomalous (1)
        """
        mocker.patch(
            "anomalib.data.utils.image.Image.open",
            return_value=Image.fromarray(np.array([[0, 236], [255, 0]], dtype=np.uint8)),
        )
        result = read_mask("dummy_path.png", as_tensor=True)

        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.uint8
        assert torch.equal(result, torch.tensor([[0, 1], [1, 0]], dtype=torch.uint8))
