"""Unit tests for BufferListMixin module."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

from anomalib.models.components.base.buffer_list import BufferListMixin


def tensorlists_are_equal(tensorlist1: list[torch.Tensor], tensorlist2: list[torch.Tensor]) -> None:
    """Check if two lists of tensors are equal."""
    if len(tensorlist1) != len(tensorlist2):
        return False
    return all((tensor1 == tensor2).all() for tensor1, tensor2 in zip(tensorlist1, tensorlist2, strict=True))


class BufferListModule(BufferListMixin):
    """Dummy module that uses the BufferListMixin to register a list of tensors as buffers."""

    def __init__(self) -> None:
        super().__init__()
        tensorlist = [torch.empty(3) for _ in range(3)]
        self.register_bufferlist("tensorlist", tensorlist)
        self.register_bufferlist("non_persistent_tensorlist", tensorlist, persistent=False)


@pytest.fixture()
def module() -> BufferListModule:
    """Fixture that returns a BufferListModule object."""
    return BufferListModule()


class TestBufferListMixin:
    """Test the BufferListMixin module."""

    def test_get_bufferlist(self, module: BufferListModule) -> None:
        """Test retrieving the tensorlist."""
        assert isinstance(module.tensorlist, list)
        assert all(isinstance(tensor, torch.Tensor) for tensor in module.tensorlist)

    def test_set_bufferlist(self, module: BufferListModule) -> None:
        """Test setting/updating the tensorlist."""
        tensorlist = [torch.rand(3) for _ in range(3)]
        module.tensorlist = tensorlist
        assert tensorlists_are_equal(module.tensorlist, tensorlist)

    def test_bufferlist_device_placement(self, module: BufferListModule) -> None:
        """Test if the device of the bufferlist is updated with the module."""
        module.cuda()
        assert all(tensor.is_cuda for tensor in module.tensorlist)
        module.cpu()
        assert all(tensor.is_cpu for tensor in module.tensorlist)

    def test_persistent_bufferlist(self) -> None:
        """Test if the bufferlist is persistent when re-loading the state dict."""
        # create a module, assign the bufferlist and get the state dict
        module = BufferListModule()
        tensorlist = [torch.rand(3) for _ in range(3)]
        module.tensorlist = tensorlist
        state_dict = module.state_dict()
        # create a new module and load the state dict
        module = BufferListModule()
        module.load_state_dict(state_dict)
        # assert that the previously set bufferlist has been restored
        assert tensorlists_are_equal(module.tensorlist, tensorlist)

    def test_non_persistent_bufferlist(self) -> None:
        """Test if the bufferlist is persistent when re-loading the state dict."""
        # create a module, assign the bufferlist and get the state dict
        module = BufferListModule()
        tensorlist = [torch.rand(3) for _ in range(3)]
        module.non_persistent_tensorlist = tensorlist
        state_dict = module.state_dict()
        # create a new module and load the state dict
        module = BufferListModule()
        module.load_state_dict(state_dict)
        # assert that the previously set bufferlist has not been restored
        assert not tensorlists_are_equal(module.non_persistent_tensorlist, tensorlist)
