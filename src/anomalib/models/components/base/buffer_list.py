"""Buffer List Mixin."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn


class BufferListMixin(nn.Module):
    """Buffer List Mixin.

    This mixin is used to allow registering a list of tensors as buffers in a pytorch module.

    Example:
        >>> class MyModule(BufferListMixin, nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         tensor_list = [torch.ones(3) * i for i in range(3)]
        ...         self.register_buffer_list("my_buffer_list", tensor_list)
        >>> module = MyModule()
        >>> # The buffer list can be accessed as a regular attribute
        >>> module.my_buffer_list
        [
            tensor([0., 0., 0.]),
            tensor([1., 1., 1.]),
            tensor([2., 2., 2.])
        ]
        >>> # We can update the buffer list at any time
        >>> new_tensor_list = [torch.ones(3) * i + 10 for i in range(3)]
        >>> module.register_buffer_list("my_buffer_list", new_tensor_list)
        >>> module.my_buffer_list
        [
            tensor([10., 10., 10.]),
            tensor([11., 11., 11.]),
            tensor([12., 12., 12.])
        ]
        >>> # Move to GPU. Since the tensors are registered as buffers, device placement is handled automatically
        >>> module.cuda()
        >>> module.my_buffer_list
        [
            tensor([10., 10., 10.], device='cuda:0'),
            tensor([11., 11., 11.], device='cuda:0'),
            tensor([12., 12., 12.], device='cuda:0')
        ]
    """

    def register_buffer_list(self, name: str, values: list[torch.Tensor], persistent: bool = True, **kwargs) -> None:
        """Register a list of tensors as buffers in a pytorch module.

        Each tensor is registered as a buffer with the name `_name_i` where `i` is the index of the tensor in the list.
        To update and retrieve the list of tensors, we dynamically assign a descriptor attribute to the class.

        Args:
            name (str): Name of the buffer list.
            values (list[torch.Tensor]): List of tensors to register as buffers.
            persistent (bool, optional): Whether the buffers should be saved as part of the module state_dict.
                Defaults to True.
            **kwargs: Additional keyword arguments to pass to `torch.nn.Module.register_buffer`.
        """
        for i, value in enumerate(values):
            self.register_buffer(f"_{name}_{i}", value, persistent=persistent, **kwargs)

        setattr(BufferListMixin, name, BufferListDescriptor(name, len(values)))


class BufferListDescriptor:
    """Buffer List Descriptor.

    This descriptor is used to allow registering a list of tensors as buffers in a pytorch module.

    Args:
        name (str): Name of the buffer list.
        length (int): Length of the buffer list.
    """

    def __init__(self, name: str, length: int) -> None:
        self.name = name
        self.length = length

    def __get__(self, instance: object, object_type: type | None = None) -> list[torch.Tensor]:
        """Get the list of tensors.

        Each element of the buffer list is stored as a buffer with the name `name_i` where `i` is the index of the
        element in the list. We use list comprehension to retrieve the list of tensors.

        Args:
            instance (object): Instance of the class.
            object_type (Any, optional): Type of the class. Defaults to None.

        Returns:
            list[torch.Tensor]: Contents of the buffer list.
        """
        del object_type
        return [getattr(instance, f"_{self.name}_{i}") for i in range(self.length)]

    def __set__(self, instance: object, values: list[torch.Tensor]) -> None:
        """Set the list of tensors.

        Assigns a new list of tensors to the buffer list by updating the individual buffer attributes.

        Args:
            instance (object): Instance of the class.
            values (list[torch.Tensor]): List of tensors to set.
        """
        for i, value in enumerate(values):
            setattr(instance, f"_{self.name}_{i}", value)
