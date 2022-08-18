"""Basic threshold where the threshold is the maximum value of the predicted anomaly scores in the validation step."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from .base import BaseThreshold


class MaximumThreshold(BaseThreshold):
    """Maximum threshold metric."""

    def update(self, preds: Tensor, _target: Tensor) -> None:
        """Update the metric."""
        self.value = torch.tensor(preds.max())

    def compute(self):
        """Return value."""
        return self.value
