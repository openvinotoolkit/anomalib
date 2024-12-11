"""Test Numpy Depth Validators."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from anomalib.data.validators.numpy.depth import NumpyDepthBatchValidator, NumpyDepthValidator


class TestNumpyDepthValidator:
    """Test NumpyDepthValidator."""

    def setup_method(self) -> None:
        """Set up the validator for each test method."""
        self.validator = NumpyDepthValidator()

    def test_validate_depth_map_valid(self) -> None:
        """Test validation of a valid depth map."""
        depth_map = np.zeros((224, 224), dtype=np.float32)
        validated_depth_map = self.validator.validate_depth_map(depth_map)
        assert isinstance(validated_depth_map, np.ndarray)
        assert validated_depth_map.shape == (224, 224)
        assert validated_depth_map.dtype == np.float32

    def test_validate_depth_map_invalid_type(self) -> None:
        """Test validation of a depth map with invalid type."""
        with pytest.raises(TypeError, match="Depth map must be a numpy array"):
            self.validator.validate_depth_map([1, 2, 3])

    def test_validate_depth_map_invalid_dimensions(self) -> None:
        """Test validation of a depth map with invalid dimensions."""
        with pytest.raises(ValueError, match="Depth map with 3 dimensions must have 1 channel, got 2."):
            self.validator.validate_depth_map(np.zeros((224, 224, 2)))

    def test_validate_depth_map_3d_valid(self) -> None:
        """Test validation of a valid 3D depth map."""
        depth_map = np.zeros((224, 224, 1), dtype=np.float32)
        validated_depth_map = self.validator.validate_depth_map(depth_map)
        assert isinstance(validated_depth_map, np.ndarray)
        assert validated_depth_map.shape == (224, 224, 1)
        assert validated_depth_map.dtype == np.float32

    def test_validate_depth_map_3d_invalid(self) -> None:
        """Test validation of an invalid 3D depth map."""
        with pytest.raises(ValueError, match="Depth map with 3 dimensions must have 1 channel"):
            self.validator.validate_depth_map(np.zeros((224, 224, 3)))

    def test_validate_depth_path_valid(self) -> None:
        """Test validation of a valid depth path."""
        depth_path = "/path/to/depth.png"
        validated_path = self.validator.validate_depth_path(depth_path)
        assert validated_path == depth_path

    def test_validate_depth_path_none(self) -> None:
        """Test validation of a None depth path."""
        assert self.validator.validate_depth_path(None) is None


class TestNumpyDepthBatchValidator:
    """Test NumpyDepthBatchValidator."""

    def setup_method(self) -> None:
        """Set up the validator for each test method."""
        self.validator = NumpyDepthBatchValidator()

    def test_validate_depth_map_valid(self) -> None:
        """Test validation of a valid depth map batch."""
        depth_map_batch = np.zeros((32, 224, 224), dtype=np.float32)
        validated_batch = self.validator.validate_depth_map(depth_map_batch)
        assert isinstance(validated_batch, np.ndarray)
        assert validated_batch.shape == (32, 224, 224)
        assert validated_batch.dtype == np.float32

    def test_validate_depth_map_invalid_type(self) -> None:
        """Test validation of a depth map batch with invalid type."""
        with pytest.raises(TypeError, match="Depth map batch must be a numpy array"):
            self.validator.validate_depth_map([1, 2, 3])

    def test_validate_depth_map_invalid_dimensions(self) -> None:
        """Test validation of a depth map batch with invalid dimensions."""
        with pytest.raises(ValueError, match="Depth map batch must have shape"):
            self.validator.validate_depth_map(np.zeros((32, 224)))

    def test_validate_depth_map_4d_valid(self) -> None:
        """Test validation of a valid 4D depth map batch."""
        depth_map_batch = np.zeros((32, 224, 224, 1), dtype=np.float32)
        validated_batch = self.validator.validate_depth_map(depth_map_batch)
        assert isinstance(validated_batch, np.ndarray)
        assert validated_batch.shape == (32, 224, 224, 1)
        assert validated_batch.dtype == np.float32

    def test_validate_depth_map_4d_invalid(self) -> None:
        """Test validation of an invalid 4D depth map batch."""
        with pytest.raises(ValueError, match="Depth map batch with 4 dimensions must have 1 channel"):
            self.validator.validate_depth_map(np.zeros((32, 224, 224, 3)))

    def test_validate_depth_path_valid(self) -> None:
        """Test validation of valid depth paths."""
        depth_paths = ["/path/to/depth1.png", "/path/to/depth2.png"]
        validated_paths = self.validator.validate_depth_path(depth_paths)
        assert validated_paths == depth_paths

    def test_validate_depth_path_none(self) -> None:
        """Test validation of None depth paths."""
        assert self.validator.validate_depth_path(None) is None

    def test_validate_depth_path_invalid_type(self) -> None:
        """Test validation of depth paths with invalid type."""
        with pytest.raises(TypeError, match="Depth path must be a list of strings"):
            self.validator.validate_depth_path("not_a_list")
