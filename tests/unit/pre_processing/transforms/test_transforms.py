"""Test transforms."""


import numpy as np
import torch

from anomalib.pre_processing.transforms import Denormalize, ToNumpy


def test_denormalize_channel_dims() -> None:
    """Test if tensor of shape (B, C, H, W) is converted to (H, W, C)."""
    tensor = torch.rand(1, 3, 4, 4)
    denormalize = Denormalize()
    result = denormalize(tensor)
    assert result.shape == (4, 4, 3)


def test_denormalize_values() -> None:
    """Test if tensor values are denormalized."""
    tensor = torch.ones(3, 10, 10)
    mean = [0.1, 0.1, 0.1]
    std = [0.5, 0.5, 0.5]
    denormalize = Denormalize(mean, std)
    result = denormalize(tensor)
    assert (result == np.full((10, 10, 3), 153, dtype=np.uint8)).all()


def test_convert_tensor_shape_1hw_to_numpy_shape_hw() -> None:
    """Convert tensor with shape (1, H, W) to numpy array with shape (H, W)."""
    # Arrange
    tensor = torch.rand(1, 4, 5)
    to_numpy = ToNumpy()

    # Act
    result = to_numpy(tensor)

    # Assert
    assert result.shape == (4, 5)


def test_convert_tensor_shape_nchw_to_numpy_array() -> None:
    """Convert tensor with shape (N, C, H, W) to numpy array with shape (N, H, W, C)."""
    # Arrange
    tensor = torch.rand(2, 3, 4, 5)
    to_numpy = ToNumpy()

    # Act
    result = to_numpy(tensor)

    # Assert
    assert result.shape == (2, 4, 5, 3)
