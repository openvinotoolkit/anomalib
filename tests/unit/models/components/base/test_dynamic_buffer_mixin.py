"""Unit tests for DynamicBufferMixin."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from anomalib.models.components.base.dynamic_buffer import DynamicBufferMixin


class DynamicBufferModule(DynamicBufferMixin):
    """Dummy module that uses the DynamicBufferMixin."""

    def __init__(self) -> None:
        super().__init__()
        self.tensor_attribute = torch.randn(4, 5)
        self.non_tensor_attribute = "Some non-tensor attribute"
        self.register_buffer("first_buffer", torch.zeros(1, 1))
        self.register_buffer("second_buffer", torch.zeros(2, 2))
        self.register_buffer("third_buffer", torch.zeros(3, 3))

    def load_from_state_dict(self, state_dict: dict, prefix: str) -> None:
        """Wrapper for the load_from_state_dict method with default args."""
        default_args = {
            "local_metadata": {},
            "strict": False,
            "missing_keys": [],
            "unexpected_keys": [],
            "error_msgs": [],
        }
        super()._load_from_state_dict(state_dict, prefix, *default_args.values())


@pytest.fixture()
def module() -> DynamicBufferModule:
    """Fixture that returns a DynamicBufferModule object."""
    return DynamicBufferModule()


class TestDynamicBufferMixin:
    """Test the DynamicBufferMixin."""

    @staticmethod
    def test_get_tensor_attribute_tensor(module: DynamicBufferModule) -> None:
        """Test the get_tensor_attribute method with a tensor field."""
        tensor_attribute = module.get_tensor_attribute("tensor_attribute")
        assert isinstance(tensor_attribute, torch.Tensor)
        assert torch.equal(tensor_attribute, module.tensor_attribute)

    @staticmethod
    def test_get_tensor_attribute_non_tensor(module: DynamicBufferModule) -> None:
        """Test the get_tensor_attribute method with a non-tensor field."""
        with pytest.raises(ValueError, match="Attribute with name 'non_tensor_attribute' is not a torch Tensor"):
            module.get_tensor_attribute("non_tensor_attribute")

    @staticmethod
    def test_load_from_state_dict(module: DynamicBufferModule) -> None:
        """Test updating the buffers from a state_dict."""
        state_dict = {
            "prefix_first_buffer": torch.zeros(5, 5),
            "prefix_second_buffer": torch.zeros(2, 2),
        }
        module.load_from_state_dict(state_dict, "prefix_")
        # buffer was in the state_dict, and shape was resized
        assert module.get_buffer("first_buffer").shape == (5, 5)
        # buffer was in the state_dict, and shape is the same
        assert module.get_buffer("second_buffer").shape == (2, 2)
        # buffer wasn't in the state_dict, and no operations were applied
        assert module.get_buffer("third_buffer").shape == (3, 3)
