"""Module that tracks the min and max values of the observations in each batch."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import torch
from torch import Tensor
from torchmetrics import Metric


class MinMax(Metric):
    """Track the min and max values of the observations in each batch."""

    full_state_update: bool = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("min", torch.tensor(float("inf")), persistent=True)
        self.add_state("max", torch.tensor(float("-inf")), persistent=True)

        self.min = torch.tensor(float("inf"))
        self.max = torch.tensor(float("-inf"))

    def update(self, predictions: Tensor, *args, **kwargs) -> None:
        """Update the min and max values."""
        del args, kwargs  # These variables are not used.

        self.max = torch.max(self.max, torch.max(predictions))
        self.min = torch.min(self.min, torch.min(predictions))

    def compute(self) -> tuple[Tensor, Tensor]:
        """Return min and max values."""
        return self.min, self.max
