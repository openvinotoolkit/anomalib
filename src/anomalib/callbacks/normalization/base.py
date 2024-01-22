"""Base Normalization Callback."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from lightning.pytorch import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib.models.components import AnomalyModule


class NormalizationCallback(Callback, ABC):
    """Base normalization callback."""

    @staticmethod
    @abstractmethod
    def _normalize_batch(batch: STEP_OUTPUT, pl_module: AnomalyModule) -> None:
        """Normalize an output batch.

        Args:
            batch (dict[str, torch.Tensor]): Output batch.
            pl_module (AnomalyModule): AnomalyModule instance.

        Returns:
            dict[str, torch.Tensor]: Normalized batch.
        """
        raise NotImplementedError
