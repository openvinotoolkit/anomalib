"""Dynamic Buffer Mixin."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC

import torch
from torch import nn


class DynamicBufferMixin(nn.Module, ABC):
    """This mixin allows loading variables from the state dict even in the case of shape mismatch."""

    def get_tensor_attribute(self, attribute_name: str) -> torch.Tensor:
        """Get attribute of the tensor given the name.

        Args:
            attribute_name (str): Name of the tensor

        Raises:
            ValueError: `attribute_name` is not a torch Tensor

        Returns:
            Tensor: torch.Tensor attribute
        """
        attribute = getattr(self, attribute_name)
        if isinstance(attribute, torch.Tensor):
            return attribute

        msg = f"Attribute with name '{attribute_name}' is not a torch Tensor"
        raise ValueError(msg)

    def _load_from_state_dict(self, state_dict: dict, prefix: str, *args) -> None:
        """Resizes the local buffers to match those stored in the state dict.

        Overrides method from parent class.

        Args:
          state_dict (dict): State dictionary containing weights
          prefix (str): Prefix of the weight file.
          *args: Variable length argument list.
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
