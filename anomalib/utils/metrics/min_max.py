"""Module that tracks the min and max values of the observations in each batch."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import torch
from torch import Tensor
from torchmetrics import Metric


class MinMax(Metric):
    """Track the min and max values of the observations in each batch."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("min", torch.tensor(float("inf")), persistent=True)  # pylint: disable=not-callable
        self.add_state("max", torch.tensor(float("-inf")), persistent=True)  # pylint: disable=not-callable

        self.min = torch.tensor(float("inf"))  # pylint: disable=not-callable
        self.max = torch.tensor(float("-inf"))  # pylint: disable=not-callable

    # pylint: disable=arguments-differ
    def update(self, predictions: Tensor) -> None:  # type: ignore
        """Update the min and max values."""
        self.max = torch.max(self.max, torch.max(predictions))
        self.min = torch.min(self.min, torch.min(predictions))

    def compute(self) -> Tuple[Tensor, Tensor]:
        """Return min and max values."""
        return self.min, self.max
