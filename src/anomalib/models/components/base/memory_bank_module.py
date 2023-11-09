"""Dynamic Buffer Module."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from abc import ABC, abstractmethod

from .anomaly_module import AnomalyModule
from .dynamic_module import DynamicBufferModule


class MemoryBankTorchModule(DynamicBufferModule, ABC):
    """Memory Bank Torch Module.

    This module is used to implement memory bank modules.
    It is a wrapper around a torch module that adds
        (i) a property to check if the module is fitted.
        (ii) a fit method that is called to fit the model using the embedding.

    The reason why anomalib need this module is because the memory bank-based models first collect the embeddings.
    Then they fit the model on the embeddings.

    This is different from the usual training process where the model is trained on batches of data.
    Hence, we need a separate module to fit the model on the embeddings.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Property to check if the model is fitted."""
        return self._is_fitted

    @abstractmethod
    def fit(self, *args, **kwargs) -> None:
        """Fit the model to the data.

        Raises:
            NotImplementedError: When the method is not implemented.
        """
        msg = f"fit method not implemented for {self.__class__.__name__}. To use a memory-bank module, implement fit."
        raise NotImplementedError(msg)


class MemoryBankAnomalyModule(AnomalyModule, ABC):
    """Memory Bank Lightning Module.

    This module is used to implement memory bank lightning modules.
    It checks if the model is fitted before validation starts.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def on_validation_start(self) -> None:
        """Ensure that the model is fitted before validation starts."""
        msg = "To use a memory-bank module, model ``fit`` must be called before validation starts."
        raise NotImplementedError(msg)
