"""Test Numpy Image Validators."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from anomalib.data.validators.numpy.image import NumpyImageBatchValidator, NumpyImageValidator


class TestNumpyImageValidator:
    """Test NumpyImageValidator."""

    def setup_method(self) -> None:
        """Set up the validator for each test method."""
        self.validator = NumpyImageValidator()

    def test_validate_image_valid(self) -> None:
        """Test validation of a valid image."""
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        validated_image = self.validator.validate_image(image)
        assert isinstance(validated_image, np.ndarray)
        assert validated_image.shape == (224, 224, 3)
        assert validated_image.dtype == np.float32
        np.testing.assert_array_equal(validated_image, image.astype(np.float32))

    def test_validate_image_invalid_type(self) -> None:
        """Test validation of an image with invalid type."""
        with pytest.raises(TypeError, match="Image must be a numpy.ndarray, got <class 'list'>"):
            self.validator.validate_image([1, 2, 3])

    def test_validate_image_adds_channel_dimension(self) -> None:
        """Test validation of an image without channel dimension."""
        # Create a 2D image without channel dimension
        input_image = np.zeros((224, 224))

        # Validate the image
        validated_image = self.validator.validate_image(input_image)
        # Check if channel dimension is added
        assert validated_image.shape == (224, 224, 1), "Channel dimension should be added"
        # Ensure the dtype is converted to float32
        assert validated_image.dtype == np.float32, "Image should be converted to float32"
        # Verify that the image content is preserved
        assert pytest.approx(validated_image[:, :, 0]) == input_image.astype(
            np.float32,
        ), "Image content should be preserved"

    def test_validate_image_invalid_channels(self) -> None:
        """Test validation of an image with invalid number of channels."""
        with pytest.raises(ValueError, match="Image must have 1 or 3 channels"):
            self.validator.validate_image(np.zeros((224, 224, 2)))

    def test_validate_image_valid_single_channel(self) -> None:
        """Test validation of a valid single-channel image."""
        image = np.zeros((224, 224, 1), dtype=np.uint8)
        validated_image = self.validator.validate_image(image)
        assert isinstance(validated_image, np.ndarray)
        assert validated_image.shape == (224, 224, 1)
        assert validated_image.dtype == np.float32

    def test_validate_gt_label_valid(self) -> None:
        """Test validation of a valid ground truth label."""
        label = 1
        validated_label = self.validator.validate_gt_label(label)
        assert isinstance(validated_label, np.ndarray)
        assert validated_label.dtype == bool
        assert validated_label.item() is True  # Use .item() to compare the scalar value

    def test_validate_gt_label_none(self) -> None:
        """Test validation of a None ground truth label."""
        assert self.validator.validate_gt_label(None) is None

    def test_validate_gt_label_invalid_type(self) -> None:
        """Test validation of a ground truth label with invalid type."""
        with pytest.raises(TypeError, match="Ground truth label must be an integer or a numpy.ndarray"):
            self.validator.validate_gt_label("1")

    def test_validate_gt_label_invalid_shape(self) -> None:
        """Test validation of a ground truth label with invalid shape."""
        with pytest.raises(ValueError, match="Ground truth label must be a scalar"):
            self.validator.validate_gt_label(np.array([0, 1]))

    def test_validate_gt_mask_valid(self) -> None:
        """Test validation of a valid ground truth mask."""
        mask = np.zeros((224, 224), dtype=np.uint8)
        validated_mask = self.validator.validate_gt_mask(mask)
        assert isinstance(validated_mask, np.ndarray)
        assert validated_mask.shape == (224, 224)
        assert validated_mask.dtype == bool

    def test_validate_gt_mask_none(self) -> None:
        """Test validation of a None ground truth mask."""
        assert self.validator.validate_gt_mask(None) is None

    def test_validate_gt_mask_invalid_type(self) -> None:
        """Test validation of a ground truth mask with invalid type."""
        with pytest.raises(TypeError, match="Mask must be a numpy.ndarray"):
            self.validator.validate_gt_mask([1, 2, 3])

    def test_validate_gt_mask_invalid_shape(self) -> None:
        """Test validation of a ground truth mask with invalid shape."""
        with pytest.raises(ValueError, match="Mask must have 1 channel, got 2."):
            self.validator.validate_gt_mask(np.zeros((224, 224, 2)))


class TestNumpyImageBatchValidator:
    """Test NumpyImageBatchValidator."""

    def setup_method(self) -> None:
        """Set up the validator for each test method."""
        self.validator = NumpyImageBatchValidator()

    def test_validate_image_valid(self) -> None:
        """Test validation of a valid image batch."""
        image_batch = np.zeros((32, 224, 224, 3), dtype=np.uint8)
        validated_batch = self.validator.validate_image(image_batch)
        assert isinstance(validated_batch, np.ndarray)
        assert validated_batch.shape == (32, 224, 224, 3)
        assert validated_batch.dtype == np.float32

    def test_validate_image_adds_channel_dimension(self) -> None:
        """Test validation of an image batch without channel dimension."""
        # Create a 3D image batch without channel dimension
        input_batch = np.zeros((32, 224, 224))

        # Validate the image batch
        validated_batch = self.validator.validate_image(input_batch)
        # Check if channel dimension is added
        assert validated_batch.shape == (32, 224, 224, 1), "Channel dimension should be added"
        # Ensure the dtype is converted to float32
        assert validated_batch.dtype == np.float32, "Image batch should be converted to float32"
        # Verify that the image content is preserved
        assert pytest.approx(validated_batch[:, :, :, 0]) == input_batch.astype(
            np.float32,
        ), "Image content should be preserved"

    def test_validate_image_invalid_type(self) -> None:
        """Test validation of an image batch with invalid type."""
        with pytest.raises(TypeError, match="Image batch must be a numpy.ndarray, got <class 'list'>"):
            self.validator.validate_image([1, 2, 3])

    def test_validate_image_adds_batch_dimension(self) -> None:
        """Test validation of an image without batch dimension."""
        # Create a 3D image without batch dimension
        input_image = np.zeros((224, 224, 3))

        # Validate the image
        validated_image = self.validator.validate_image(input_image)

        # Check if batch dimension is added
        assert validated_image.shape == (1, 224, 224, 3), "Batch dimension should be added"
        # Ensure the dtype is converted to float32
        assert validated_image.dtype == np.float32, "Image should be converted to float32"
        # Verify that the image content is preserved
        assert np.array_equal(validated_image[0], input_image.astype(np.float32)), "Image content should be preserved"

    def test_validate_image_invalid_channels(self) -> None:
        """Test validation of an image batch with invalid number of channels."""
        with pytest.raises(ValueError, match="Image batch must have 1 or 3 channels"):
            self.validator.validate_image(np.zeros((32, 224, 224, 2)))

    def test_validate_image_valid_single_channel(self) -> None:
        """Test validation of a valid single-channel image batch."""
        image_batch = np.zeros((32, 224, 224, 1), dtype=np.uint8)
        validated_batch = self.validator.validate_image(image_batch)
        assert isinstance(validated_batch, np.ndarray)
        assert validated_batch.shape == (32, 224, 224, 1)
        assert validated_batch.dtype == np.float32

    def test_validate_gt_label_valid(self) -> None:
        """Test validation of valid ground truth labels."""
        labels = np.array([0, 1, 1, 0])
        validated_labels = self.validator.validate_gt_label(labels)
        assert isinstance(validated_labels, np.ndarray)
        assert validated_labels.dtype == bool
        assert np.array_equal(validated_labels, np.array([False, True, True, False]))

    def test_validate_gt_label_none(self) -> None:
        """Test validation of None ground truth labels."""
        assert self.validator.validate_gt_label(None) is None

    def test_validate_gt_label_valid_sequence(self) -> None:
        """Test validation of ground truth labels with sequence input."""
        # Test with binary labels (0: normal, 1: anomaly)
        validated_labels = self.validator.validate_gt_label([0, 1, 1, 0])
        assert isinstance(validated_labels, np.ndarray)
        assert validated_labels.dtype == bool
        assert np.array_equal(validated_labels, np.array([False, True, True, False]))

    def test_validate_gt_label_invalid_dimensions(self) -> None:
        """Test validation of ground truth labels with invalid dimensions."""
        with pytest.raises(ValueError, match="Ground truth label batch must be 1-dimensional"):
            self.validator.validate_gt_label(np.array([[0, 1], [1, 0]]))

    def test_validate_gt_mask_valid(self) -> None:
        """Test validation of valid ground truth masks."""
        masks = np.zeros((4, 224, 224), dtype=np.uint8)
        validated_masks = self.validator.validate_gt_mask(masks)
        assert isinstance(validated_masks, np.ndarray)
        assert validated_masks.shape == (4, 224, 224)
        assert validated_masks.dtype == bool

    def test_validate_gt_mask_none(self) -> None:
        """Test validation of None ground truth masks."""
        assert self.validator.validate_gt_mask(None) is None

    def test_validate_gt_mask_invalid_type(self) -> None:
        """Test validation of ground truth masks with invalid type."""
        with pytest.raises(TypeError, match="Ground truth mask batch must be a numpy.ndarray"):
            self.validator.validate_gt_mask([np.zeros((224, 224))])

    def test_validate_gt_mask_invalid_dimensions(self) -> None:
        """Test validation of ground truth masks with invalid dimensions."""
        with pytest.raises(ValueError, match="Ground truth mask batch must have 1 channel, got 224"):
            self.validator.validate_gt_mask(np.zeros((4, 224, 224, 224)))
