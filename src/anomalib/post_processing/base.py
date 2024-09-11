"""Base class for post-processor."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from lightning.pytorch import Callback
from torch import nn

from anomalib.data import InferenceBatch


class PostProcessor(nn.Module, Callback, ABC):
    """Base class for post-processor.

    The post-processor is a callback that is used to post-process the predictions of the model.
    """

    @abstractmethod
    def forward(self, batch: InferenceBatch) -> InferenceBatch:
        """Functional forward method for post-processing."""
