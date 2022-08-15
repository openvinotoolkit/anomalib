"""Basic threshold where the threshold is the maximum value of the predicted anomaly scores in the validation step."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from torch import Tensor

from .base import BaseThreshold


class MaximumThreshold(BaseThreshold):
    """Maximum threshold metric."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(0.0, **kwargs)

    def update(self, preds: Tensor, _target: Tensor) -> None:
        """Update the metric."""
        self.value = torch.tensor(preds.max())

    def compute(self):
        """Return value."""
        return self.value
