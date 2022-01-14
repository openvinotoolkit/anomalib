"""Module that tracks the min and max values of the observations in each batch."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

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
