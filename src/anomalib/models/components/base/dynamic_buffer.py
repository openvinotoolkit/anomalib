"""Dynamic Buffer Mixin.

This mixin class enables loading state dictionaries with mismatched tensor shapes
by dynamically resizing buffers to match the loaded state.

Example:
    >>> import torch
    >>> from torch import nn
    >>> class MyModule(DynamicBufferMixin, nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.register_buffer("buffer", torch.ones(3))
    ...
    >>> module = MyModule()
    >>> # Original buffer shape is (3,)
    >>> module.buffer
    tensor([1., 1., 1.])
    >>> # Load state dict with different buffer shape (5,)
    >>> new_state = {"buffer": torch.ones(5)}
    >>> module.load_state_dict(new_state)
    >>> # Buffer is automatically resized
    >>> module.buffer
    tensor([1., 1., 1., 1., 1.])
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC

import torch
from torch import nn


class DynamicBufferMixin(nn.Module, ABC):
    """Mixin that enables loading state dicts with mismatched tensor shapes.

    This mixin class extends ``nn.Module`` to allow loading state dictionaries
    even when the shapes of tensors in the state dict do not match the shapes
    of the module's buffers. When loading a state dict, the mixin automatically
    resizes any mismatched buffers to match the shapes in the state dict.
    """

    def get_tensor_attribute(self, attribute_name: str) -> torch.Tensor:
        """Get a tensor attribute by name.

        Args:
            attribute_name (str): Name of the tensor attribute to retrieve

        Raises:
            ValueError: If the attribute with name ``attribute_name`` is not a
                ``torch.Tensor``

        Returns:
            torch.Tensor: The tensor attribute
        """
        attribute = getattr(self, attribute_name)
        if isinstance(attribute, torch.Tensor):
            return attribute

        msg = f"Attribute with name '{attribute_name}' is not a torch Tensor"
        raise ValueError(msg)

    def _load_from_state_dict(self, state_dict: dict, prefix: str, *args) -> None:
        """Load a state dictionary, resizing buffers if shapes don't match.

        This method overrides the parent class implementation to handle tensor
        shape mismatches when loading state dictionaries. It resizes any local
        buffers whose shapes don't match the corresponding tensors in the state
        dict.

        Args:
            state_dict (dict): Dictionary containing state to load
            prefix (str): Prefix to prepend to parameter/buffer names
            *args: Additional arguments passed to parent implementation
        """
        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_buffers = {k: v for k, v in persistent_buffers.items() if v is not None}

        for param in local_buffers:
            for key in state_dict:
                if (
                    key.startswith(prefix)
                    and key[len(prefix) :].split(".")[0] == param
                    and local_buffers[param].shape != state_dict[key].shape
                ):
                    attribute = self.get_tensor_attribute(param)
                    attribute.resize_(state_dict[key].shape)

        super()._load_from_state_dict(state_dict, prefix, *args)
