"""Memory Bank Module."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod

import torch
from torch import nn


class MemoryBankMixin(nn.Module):
    """Memory Bank Lightning Module.

    This module is used to implement memory bank lightning modules.
    It checks if the model is fitted before validation starts.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer("_is_fitted", torch.tensor([False]))
        self._is_fitted: torch.Tensor

    @abstractmethod
    def fit(self) -> None:
        """Fit the model to the data."""
        msg = (
            f"fit method not implemented for {self.__class__.__name__}. "
            "To use a memory-bank module, implement ``fit.``"
        )
        raise NotImplementedError(msg)

    def on_validation_start(self) -> None:
        """Ensure that the model is fitted before validation starts."""
        if not self._is_fitted:
            self.fit()
            self._is_fitted = torch.tensor([True])

    def on_train_epoch_end(self) -> None:
        """Ensure that the model is fitted before validation starts."""
        if not self._is_fitted:
            self.fit()
            self._is_fitted = torch.tensor([True])
