"""Module that tracks the min and max values of the observations in each batch.

This module provides the ``MinMax`` metric class which tracks the minimum and
maximum values seen across batches of data. This is useful for normalizing
predictions or monitoring value ranges during training.

Example:
    >>> from anomalib.metrics import MinMax
    >>> import torch
    >>> # Create sample predictions
    >>> predictions = torch.tensor([0.0807, 0.6329, 0.0559, 0.9860, 0.3595])
    >>> # Initialize and compute min/max
    >>> minmax = MinMax()
    >>> min_val, max_val = minmax(predictions)
    >>> min_val, max_val
    (tensor(0.0559), tensor(0.9860))

    The metric can be updated incrementally with new batches:

    >>> new_predictions = torch.tensor([0.3251, 0.3169, 0.3072, 0.6247, 0.9999])
    >>> minmax.update(new_predictions)
    >>> min_val, max_val = minmax.compute()
    >>> min_val, max_val
    (tensor(0.0559), tensor(0.9999))
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torchmetrics import Metric


class MinMax(Metric):
    """Track minimum and maximum values across batches.

    This metric maintains running minimum and maximum values across all batches
    it processes. It is useful for tasks like normalization or monitoring the
    range of values during training.

    Args:
        full_state_update (bool, optional): Whether to update the internal state
            with each new batch. Defaults to ``True``.
        kwargs: Additional keyword arguments passed to the parent class.

    Attributes:
        min (torch.Tensor): Running minimum value seen across all batches
        max (torch.Tensor): Running maximum value seen across all batches

    Example:
        >>> from anomalib.metrics import MinMax
        >>> import torch
        >>> # Create metric
        >>> minmax = MinMax()
        >>> # Update with batches
        >>> batch1 = torch.tensor([0.1, 0.2, 0.3])
        >>> batch2 = torch.tensor([0.2, 0.4, 0.5])
        >>> minmax.update(batch1)
        >>> minmax.update(batch2)
        >>> # Get final min/max values
        >>> min_val, max_val = minmax.compute()
        >>> min_val, max_val
        (tensor(0.1000), tensor(0.5000))
    """

    full_state_update: bool = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("min", torch.tensor(float("inf")), persistent=True, dist_reduce_fx="min")
        self.add_state("max", torch.tensor(float("-inf")), persistent=True, dist_reduce_fx="max")

        self.min = torch.tensor(float("inf"))
        self.max = torch.tensor(float("-inf"))

    def update(self, predictions: torch.Tensor, *args, **kwargs) -> None:
        """Update running min and max values with new predictions.

        Args:
            predictions (torch.Tensor): New tensor of values to include in min/max
                tracking
            *args: Additional positional arguments (unused)
            **kwargs: Additional keyword arguments (unused)
        """
        del args, kwargs  # These variables are not used.

        self.min = torch.min(self.min, torch.min(predictions))
        self.max = torch.max(self.min, torch.max(predictions))

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute final minimum and maximum values.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing the (min, max)
                values tracked across all batches
        """
        return self.min, self.max
