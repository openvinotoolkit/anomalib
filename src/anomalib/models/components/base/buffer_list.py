"""Buffer List Mixin."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import torch
from torch import nn


class BufferListMixin(nn.Module):
    """Buffer List Mixin.

    This mixin is used to allow registering a list of tensors as buffers in a pytorch module.
    """

    def register_bufferlist(self, name: str, values: list[torch.Tensor], persistent: bool = True, **kwargs) -> None:
        """Register a list of tensors as buffers in a pytorch module.

        Each tensor is registered as a buffer with the name `name_i` where `i` is the index of the tensor in the list.
        To retrieve the list of tensors, we dynamically create a property with the name `name` that returns the list of
        tensors.

        Args:
            name (str): Name of the bufferlist.
            values (list[torch.Tensor]): List of tensors to register as buffers.
            persistent (bool, optional): Whether the buffers should be saved as part of the module state_dict.
                Defaults to True.
            **kwargs: Additional keyword arguments to pass to `torch.nn.Module.register_buffer`.

        Example:
            >>> class MyModule(BufferListMixin, nn.Module):
            ...     def __init__(self):
            ...         tensor_list = [torch.ones(3) * i for i in range(3)]
            ...         self.register_bufferlist("my_bufferlist", tensor_list)
            >>> module = MyModule()
            >>> module.my_bufferlist
            [
                tensor([0., 0., 0.]),
                tensor([1., 1., 1.]),
                tensor([2., 2., 2.])
            ]
            >>> module.cuda()  # Move to GPU. Since the tensors are registered as buffers, they will be moved to GPU.
            >>> module.my_bufferlist
            [
                tensor([0., 0., 0.], device='cuda:0'),
                tensor([1., 1., 1.], device='cuda:0'),
                tensor([2., 2., 2.], device='cuda:0')
            ]
        """
        for i, value in enumerate(values):
            self.register_buffer(f"_{name}_{i}", value, persistent=persistent, **kwargs)

        setattr(BufferListMixin, name, BufferListDescriptor(name, len(values)))


class BufferListDescriptor:
    """Buffer List Descriptor.

    This descriptor is used to allow registering a list of tensors as buffers in a pytorch module.

    Args:
        name (str): Name of the bufferlist.
        length (int): Length of the bufferlist.
    """

    def __init__(self, name: str, length: int) -> None:
        self.name = name
        self.length = length

    def __get__(self, instance: object, object_type: type | None = None) -> list[torch.Tensor]:
        """Get the list of tensors.

        Each element of the bufferlist is stored as a buffer with the name `name_i` where `i` is the index of the
        element in the list. We use list comprehension to retrieve the list of tensors.

        Args:
            instance (object): Instance of the class.
            object_type (Any, optional): Type of the class. Defaults to None.

        Returns:
            list[torch.Tensor]: Contents of the bufferlist.
        """
        del object_type
        return [getattr(instance, f"_{self.name}_{i}") for i in range(self.length)]

    def __set__(self, instance: object, values: list[torch.Tensor]) -> None:
        """Set the list of tensors.

        Assigns a new list of tensors to the bufferlist by updating the individual buffer attributes.

        Args:
            instance (object): Instance of the class.
            values (list[torch.Tensor]): List of tensors to set.
        """
        for i, value in enumerate(values):
            setattr(instance, f"_{self.name}_{i}", value)
