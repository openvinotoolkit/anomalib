"""Test Numpy Video Validators."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from anomalib.data.validators.numpy.video import NumpyVideoBatchValidator, NumpyVideoValidator


class TestNumpyVideoValidator:
    """Test NumpyVideoValidator."""

    def setup_method(self) -> None:
        """Set up the validator for each test method."""
        self.validator = NumpyVideoValidator()

    def test_validate_image_valid(self) -> None:
        """Test validation of a valid image."""
        image = np.zeros((10, 224, 224, 3), dtype=np.uint8)
        validated_image = self.validator.validate_image(image)
        assert isinstance(validated_image, np.ndarray)
        assert validated_image.shape == (10, 224, 224, 3)
        assert validated_image.dtype == np.float32
        np.testing.assert_array_equal(validated_image, image.astype(np.float32))

    def test_validate_image_invalid_type(self) -> None:
        """Test validation of an image with invalid type."""
        with pytest.raises(TypeError, match="Video must be a numpy.ndarray, got <class 'list'>"):
            self.validator.validate_image([1, 2, 3])

    def test_validate_image_adds_time_dimension(self) -> None:
        """Test validation of an image without time dimension."""
        # Create a 3D image without time dimension
        input_image = np.zeros((224, 224, 3))

        # Validate the image
        validated_image = self.validator.validate_image(input_image)
        # Check if time dimension is added
        assert validated_image.shape == (1, 224, 224, 3), "Time dimension should be added"
        # Ensure the dtype is converted to float32
        assert validated_image.dtype == np.float32, "Image should be converted to float32"
        # Verify that the image content is preserved
        assert pytest.approx(validated_image[0]) == input_image.astype(np.float32), "Image content should be preserved"

    def test_validate_image_invalid_channels(self) -> None:
        """Test validation of an image with invalid number of channels."""
        with pytest.raises(ValueError, match="Video must have 1 or 3 channels"):
            self.validator.validate_image(np.zeros((10, 224, 224, 2)))

    def test_validate_image_valid_single_channel(self) -> None:
        """Test validation of a valid single-channel image."""
        image = np.zeros((10, 224, 224, 1), dtype=np.uint8)
        validated_image = self.validator.validate_image(image)
        assert isinstance(validated_image, np.ndarray)
        assert validated_image.shape == (10, 224, 224, 1)
        assert validated_image.dtype == np.float32

    def test_validate_target_frame_valid(self) -> None:
        """Test validation of a valid target frame."""
        target_frame = 5
        assert self.validator.validate_target_frame(target_frame) == target_frame

    def test_validate_target_frame_none(self) -> None:
        """Test validation of a None target frame."""
        assert self.validator.validate_target_frame(None) is None

    def test_validate_target_frame_invalid_type(self) -> None:
        """Test validation of a target frame with invalid type."""
        with pytest.raises(TypeError, match="Target frame must be an integer"):
            self.validator.validate_target_frame("5")

    def test_validate_target_frame_negative(self) -> None:
        """Test validation of a negative target frame."""
        with pytest.raises(ValueError, match="Target frame index must be non-negative"):
            self.validator.validate_target_frame(-1)

    def test_validate_gt_label_valid(self) -> None:
        """Test validation of a valid ground truth label."""
        # Test with binary label (0: normal, 1: anomaly)
        label = 1
        validated_label = self.validator.validate_gt_label(label)
        assert isinstance(validated_label, np.ndarray)
        assert validated_label.dtype == bool
        assert validated_label.item() is True


class TestNumpyVideoBatchValidator:
    """Test NumpyVideoBatchValidator."""

    def setup_method(self) -> None:
        """Set up the validator for each test method."""
        self.validator = NumpyVideoBatchValidator()

    def test_validate_image_valid(self) -> None:
        """Test validation of a valid image batch."""
        image_batch = np.zeros((2, 10, 224, 224, 3), dtype=np.uint8)
        validated_batch = self.validator.validate_image(image_batch)
        assert isinstance(validated_batch, np.ndarray)
        assert validated_batch.shape == (2, 10, 224, 224, 3)
        assert validated_batch.dtype == np.float32

    def test_validate_image_adds_time_dimension(self) -> None:
        """Test validation of an image batch without time dimension."""
        # Create a 4D image batch without time dimension
        input_batch = np.zeros((2, 224, 224, 3))

        # Validate the image batch
        validated_batch = self.validator.validate_image(input_batch)
        # Check if time dimension is added
        assert validated_batch.shape == (2, 224, 224, 3), "Time dimension should not be added for batch input"
        # Ensure the dtype is converted to float32
        assert validated_batch.dtype == np.float32, "Image batch should be converted to float32"
        # Verify that the image content is preserved
        assert pytest.approx(validated_batch) == input_batch.astype(np.float32), "Image content should be preserved"

    def test_validate_image_invalid_type(self) -> None:
        """Test validation of an image batch with invalid type."""
        with pytest.raises(TypeError, match="Video batch must be a numpy.ndarray, got <class 'list'>"):
            self.validator.validate_image([1, 2, 3])

    def test_validate_image_invalid_dimensions(self) -> None:
        """Test validation of an image batch with invalid dimensions."""
        with pytest.raises(ValueError, match="Video batch must have 4 or 5 dimensions, got shape \\(224, 224, 3\\)"):
            self.validator.validate_image(np.zeros((224, 224, 3)))

    def test_validate_image_invalid_channels(self) -> None:
        """Test validation of an image batch with invalid number of channels."""
        with pytest.raises(ValueError, match="Video batch must have 1 or 3 channels, got 2"):
            self.validator.validate_image(np.zeros((2, 10, 224, 224, 2)))

    def test_validate_image_valid_single_channel(self) -> None:
        """Test validation of a valid single-channel image batch."""
        image_batch = np.zeros((2, 10, 224, 224, 1), dtype=np.uint8)
        validated_batch = self.validator.validate_image(image_batch)
        assert isinstance(validated_batch, np.ndarray)
        assert validated_batch.shape == (2, 10, 224, 224, 1)
        assert validated_batch.dtype == np.float32

    def test_validate_gt_label_valid(self) -> None:
        """Test validation of valid ground truth labels."""
        labels = np.array([0, 1])
        validated_labels = self.validator.validate_gt_label(labels)
        assert isinstance(validated_labels, np.ndarray)
        assert validated_labels.dtype == bool
        assert np.array_equal(validated_labels, np.array([False, True]))

    def test_validate_gt_label_none(self) -> None:
        """Test validation of None ground truth labels."""
        assert self.validator.validate_gt_label(None) is None

    def test_validate_gt_label_valid_sequence(self) -> None:
        """Test validation of ground truth labels with sequence input."""
        # Test with binary labels (0: normal, 1: anomaly)
        labels = [0, 1]
        validated_labels = self.validator.validate_gt_label(labels)
        assert isinstance(validated_labels, np.ndarray)
        assert validated_labels.dtype == bool
        assert np.array_equal(validated_labels, np.array([False, True]))

    def test_validate_gt_label_invalid_type(self) -> None:
        """Test validation of ground truth labels with invalid type."""
        # Test with a non-sequence, non-array type
        with pytest.raises(TypeError, match="Ground truth label batch must be a numpy.ndarray"):
            self.validator.validate_gt_label(3.14)

    def test_validate_gt_label_invalid_dimensions(self) -> None:
        """Test validation of ground truth labels with invalid dimensions."""
        with pytest.raises(ValueError, match="Ground truth label batch must be 1-dimensional, got shape \\(2, 2\\)"):
            self.validator.validate_gt_label(np.array([[0, 1], [1, 0]]))

    def test_validate_gt_label_invalid_dtype(self) -> None:
        """Test validation of ground truth labels with invalid dtype."""
        # Test that float labels are converted to boolean
        labels = np.array([0.5, 1.5])
        validated_labels = self.validator.validate_gt_label(labels)
        assert isinstance(validated_labels, np.ndarray)
        assert validated_labels.dtype == bool
        assert np.array_equal(validated_labels, np.array([True, True]))
