"""Unit tests for BufferListMixin module."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from anomalib.models.components.base.buffer_list import BufferListMixin


def tensor_lists_are_equal(tensor_list1: list[torch.Tensor], tensor_list2: list[torch.Tensor]) -> None:
    """Check if two lists of tensors are equal."""
    if len(tensor_list1) != len(tensor_list2):
        return False
    return all((tensor1 == tensor2).all() for tensor1, tensor2 in zip(tensor_list1, tensor_list2, strict=True))


class BufferListModule(BufferListMixin):
    """Dummy module that uses the BufferListMixin to register a list of tensors as buffers."""

    def __init__(self) -> None:
        super().__init__()
        tensor_list = [torch.empty(3) for _ in range(3)]
        self.register_buffer_list("tensor_list", tensor_list)
        self.register_buffer_list("non_persistent_tensor_list", tensor_list, persistent=False)


@pytest.fixture()
def module() -> BufferListModule:
    """Fixture that returns a BufferListModule object."""
    return BufferListModule()


class TestBufferListMixin:
    """Test the BufferListMixin module."""

    @staticmethod
    def test_get_buffer_list(module: BufferListModule) -> None:
        """Test retrieving the tensor_list."""
        assert isinstance(module.tensor_list, list)
        assert all(isinstance(tensor, torch.Tensor) for tensor in module.tensor_list)

    @staticmethod
    def test_set_buffer_list(module: BufferListModule) -> None:
        """Test setting/updating the tensor_list."""
        tensor_list = [torch.rand(3) for _ in range(3)]
        module.tensor_list = tensor_list
        assert tensor_lists_are_equal(module.tensor_list, tensor_list)

    @staticmethod
    def test_buffer_list_device_placement(module: BufferListModule) -> None:
        """Test if the device of the buffer list is updated with the module."""
        module.cuda()
        assert all(tensor.is_cuda for tensor in module.tensor_list)
        module.cpu()
        assert all(tensor.is_cpu for tensor in module.tensor_list)

    @staticmethod
    def test_persistent_buffer_list(module: BufferListModule) -> None:
        """Test if the buffer_list is persistent when re-loading the state dict."""
        # create a module, assign the buffer list and get the state dict
        module = BufferListModule()
        tensor_list = [torch.rand(3) for _ in range(3)]
        module.tensor_list = tensor_list
        state_dict = module.state_dict()
        # create a new module and load the state dict
        module = BufferListModule()
        module.load_state_dict(state_dict)
        # assert that the previously set buffer list has been restored
        assert tensor_lists_are_equal(module.tensor_list, tensor_list)

    @staticmethod
    def test_non_persistent_buffer_list(module: BufferListModule) -> None:
        """Test if the buffer_list is persistent when re-loading the state dict."""
        # create a module, assign the buffer list and get the state dict
        module = BufferListModule()
        tensor_list = [torch.rand(3) for _ in range(3)]
        module.non_persistent_tensor_list = tensor_list
        state_dict = module.state_dict()
        # create a new module and load the state dict
        module = BufferListModule()
        module.load_state_dict(state_dict)
        # assert that the previously set buffer list has not been restored
        assert not tensor_lists_are_equal(module.non_persistent_tensor_list, tensor_list)
