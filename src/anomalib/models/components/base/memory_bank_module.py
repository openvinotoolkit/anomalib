"""Memory Bank Module.

This module provides a mixin class for implementing memory bank-based anomaly
detection models. Memory banks store reference features or embeddings that are
used to detect anomalies by comparing test samples against the stored references.

The mixin ensures proper initialization and fitting of the memory bank before
validation or inference.

Example:
    Create a custom memory bank model:

    >>> from anomalib.models.components.base import MemoryBankMixin
    >>> class MyMemoryModel(MemoryBankMixin):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.memory = []
    ...
    ...     def fit(self):
    ...         # Implement memory bank population logic
    ...         self.memory = [1, 2, 3]
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod

import torch
from torch import nn


class MemoryBankMixin(nn.Module):
    """Memory Bank Lightning Module.

    This mixin class provides functionality for memory bank-based models that need
    to store and compare against reference features/embeddings. It ensures the
    memory bank is properly fitted before validation or inference begins.

    The mixin tracks the fitting status via a persistent buffer ``_is_fitted``
    and automatically triggers the fitting process when needed.

    Attributes:
        device (torch.device): Device where the model/tensors reside
        _is_fitted (torch.Tensor): Boolean tensor tracking if model is fitted
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer("_is_fitted", torch.tensor([False]))
        self.device: torch.device  # defined in lightning module
        self._is_fitted: torch.Tensor

    @abstractmethod
    def fit(self) -> None:
        """Fit the memory bank model to the training data.

        This method should be implemented by subclasses to define how the memory
        bank is populated with reference features/embeddings.

        Raises:
            NotImplementedError: If the subclass does not implement this method
        """
        msg = (
            f"fit method not implemented for {self.__class__.__name__}. To use a memory-bank module, implement ``fit``."
        )
        raise NotImplementedError(msg)

    def on_validation_start(self) -> None:
        """Ensure memory bank is fitted before validation.

        This hook automatically fits the memory bank if it hasn't been fitted yet.
        """
        if not self._is_fitted:
            self.fit()
            self._is_fitted = torch.tensor([True], device=self.device)

    def on_train_epoch_end(self) -> None:
        """Ensure memory bank is fitted after training.

        This hook automatically fits the memory bank if it hasn't been fitted yet.
        """
        if not self._is_fitted:
            self.fit()
            self._is_fitted = torch.tensor([True], device=self.device)
