"""Buffer List Mixin.

This mixin allows registering a list of tensors as buffers in a PyTorch module.

Example:
    >>> # Create a module that uses the buffer list mixin
    >>> class MyModule(BufferListMixin, nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         tensor_list = [torch.ones(3) * i for i in range(3)]
    ...         self.register_buffer_list("my_buffer_list", tensor_list)
    ...
    >>> # Initialize the module
    >>> module = MyModule()
    ...
    >>> # The buffer list can be accessed as a regular attribute
    >>> module.my_buffer_list
    [
        tensor([0., 0., 0.]),
        tensor([1., 1., 1.]),
        tensor([2., 2., 2.])
    ]
    ...
    >>> # Update the buffer list with new tensors
    >>> new_tensor_list = [torch.ones(3) * i + 10 for i in range(3)]
    >>> module.register_buffer_list("my_buffer_list", new_tensor_list)
    >>> module.my_buffer_list
    [
        tensor([10., 10., 10.]),
        tensor([11., 11., 11.]),
        tensor([12., 12., 12.])
    ]
    ...
    >>> # Move to GPU - device placement is handled automatically
    >>> module.cuda()
    >>> module.my_buffer_list
    [
        tensor([10., 10., 10.], device='cuda:0'),
        tensor([11., 11., 11.], device='cuda:0'),
        tensor([12., 12., 12.], device='cuda:0')
    ]
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn


class BufferListMixin(nn.Module):
    """Mixin class that enables registering lists of tensors as module buffers.

    This mixin extends PyTorch modules to support registering lists of tensors as
    buffers, which are automatically handled during device placement and state
    dict operations.
    """

    def register_buffer_list(
        self,
        name: str,
        values: list[torch.Tensor],
        persistent: bool = True,
        **kwargs,
    ) -> None:
        """Register a list of tensors as buffers in a PyTorch module.

        Each tensor is registered as a buffer with the name ``_name_i`` where ``i``
        is the index of the tensor in the list. The list can be accessed and
        updated using the original ``name``.

        Args:
            name (str):
                Name of the buffer list.
            values (list[torch.Tensor]):
                List of tensors to register as buffers.
            persistent (bool, optional):
                Whether the buffers should be saved as part of the module
                state_dict. Defaults to ``True``.
            **kwargs:
                Additional keyword arguments to pass to
                ``torch.nn.Module.register_buffer``.
        """
        for i, value in enumerate(values):
            self.register_buffer(f"_{name}_{i}", value, persistent=persistent, **kwargs)

        setattr(BufferListMixin, name, BufferListDescriptor(name, len(values)))


class BufferListDescriptor:
    """Descriptor class for managing lists of buffer tensors.

    This descriptor provides the functionality to access and modify lists of
    tensors that are registered as buffers in a PyTorch module.

    Args:
        name (str):
            Name of the buffer list.
        length (int):
            Length of the buffer list.
    """

    def __init__(self, name: str, length: int) -> None:
        self.name = name
        self.length = length

    def __get__(
        self,
        instance: object,
        object_type: type | None = None,
    ) -> list[torch.Tensor]:
        """Get the list of tensors.

        Retrieves the list of tensors stored as individual buffers with names
        ``_name_i`` where ``i`` is the index.

        Args:
            instance (object):
                Instance of the class.
            object_type (type | None, optional):
                Type of the class. Defaults to ``None``.

        Returns:
            list[torch.Tensor]:
                List of tensor buffers.
        """
        del object_type
        return [getattr(instance, f"_{self.name}_{i}") for i in range(self.length)]

    def __set__(self, instance: object, values: list[torch.Tensor]) -> None:
        """Set the list of tensors.

        Updates the individual buffer attributes with new tensor values.

        Args:
            instance (object):
                Instance of the class.
            values (list[torch.Tensor]):
                List of tensors to set as buffers.
        """
        for i, value in enumerate(values):
            setattr(instance, f"_{self.name}_{i}", value)
