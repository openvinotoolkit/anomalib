"""
Dynamic Buffer Module
"""

from abc import ABC

import torch.nn as nn
from torch import Tensor


class DynamicBufferModule(ABC, nn.Module):
    """
    Torch module that allows loading variables from the state dict even in the case of shape mismatch.
    """

    def get_tensor_attribute(self, attribute_name: str) -> Tensor:
        """
        get_tensor [summary]

        Args:
            attribute_name (str): [description]

        Returns:
            Tensor: [description]
        """
        attribute = self.__getattr__(attribute_name)
        if isinstance(attribute, Tensor):
            return attribute

        raise ValueError(f"Attribute with name '{attribute_name}' is not a torch Tensor")

    def _load_from_state_dict(self, state_dict: dict, prefix: str, *args):
        """
        Overrides method from parent class. Resizes the local buffers to match those stored in the state dict.

        Args:
          state_dict: dict: State dictionary containing weights
          prefix: str: Prefix of the weight file.
          *args:

        """
        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_buffers = {k: v for k, v in persistent_buffers.items() if v is not None}

        for param in local_buffers.keys():
            for key in state_dict.keys():
                if key.startswith(prefix) and key[len(prefix) :].split(".")[0] == param:
                    if not local_buffers[param].shape == state_dict[key].shape:
                        attribute = self.get_tensor_attribute(param)
                        attribute.resize_(state_dict[key].shape)

        super()._load_from_state_dict(state_dict, prefix, *args)
