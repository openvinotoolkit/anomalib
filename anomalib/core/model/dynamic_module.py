import torch.nn as nn


class DynamicBufferModule(nn.Module):
    """
    Torch module that allows loading variables from the state dict even in the case of shape mismatch.
    """

    def _load_from_state_dict(self, state_dict: dict, prefix: str, *args):
        """
        overrides method from parent class. Resizes the local buffers to match those stored in the state dict.
        """
        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_buffers = {k: v for k, v in persistent_buffers.items() if v is not None}

        for param in local_buffers.keys():
            for key in state_dict.keys():
                if key.startswith(prefix) and key[len(prefix) :].split(".")[0] == param:
                    if not local_buffers[param].shape == state_dict[key].shape:
                        self.__getattr__(param).resize_(state_dict[key].shape)
        super()._load_from_state_dict(state_dict, prefix, *args)
