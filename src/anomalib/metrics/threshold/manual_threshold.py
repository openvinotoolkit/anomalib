"""Container to hold manual threshold values for image and pixel metrics."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from .base import Threshold


class ManualThreshold(Threshold):
    """Initialize Manual Threshold.

    Args:
        default_value (float, optional): Default threshold value.
            Defaults to ``0.5``.
        kwargs: Any keyword arguments.

    Examples:
        >>> from anomalib.metrics import ManualThreshold
        >>> import torch
        ...
        >>> manual_threshold = ManualThreshold(default_value=0.5)
        ...
        >>> labels = torch.randint(low=0, high=2, size=(5,))
        >>> preds = torch.rand(5)
        ...
        >>> threshold = manual_threshold(preds, labels)
        >>> threshold
        tensor(0.5000, dtype=torch.float64)

        As the threshold is manually set, the threshold value is the same as the
        ``default_value``.

        >>> labels = torch.randint(low=0, high=2, size=(5,))
        >>> preds = torch.rand(5)
        >>> threshold = manual_threshold(preds2, labels2)
        >>> threshold
        tensor(0.5000, dtype=torch.float64)

        The threshold value remains the same even if the inputs change.
    """

    def __init__(self, default_value: float = 0.5, **kwargs) -> None:
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

    @staticmethod
    def update(*args, **kwargs) -> None:
        """Do nothing.

        Args:
            *args: Any positional arguments.
            **kwargs: Any keyword arguments.
        """
        del args, kwargs  # Unused arguments.
