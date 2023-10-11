"""Container to hold manual threshold values for image and pixel metrics."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import torch

from .base import BaseThreshold


class ManualThreshold(BaseThreshold):
    def __init__(self, default_value: float = 0.5, **kwargs) -> None:
        """Initialize Manual Threshold.

        Args:
            default_value (float, optional): Default threshold value. Defaults to 0.5.
        """
        super().__init__(**kwargs)
        self.add_state("value", default=torch.tensor(default_value, dtype=torch.float64), persistent=True)
        self.value = torch.tensor(default_value, dtype=torch.float64)

    def compute(self) -> torch.Tensor:
        """Compute the threshold.

        In case of manual thresholding, the threshold is already set and does not need to be computed.

        Returns:
            torch.Tensor: Value of the optimal threshold.
        """
        return self.value

    def update(self, *args, **kwargs) -> None:
        """Do nothing.

        Args:
            *args: Any positional arguments.
            **kwargs: Any keyword arguments.
        """
        del args, kwargs  # Unused arguments.
        return
