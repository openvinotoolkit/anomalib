"""Module that tracks the min and max values of the observations in each batch."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torchmetrics import Metric


class MinMax(Metric):
    """Track the min and max values of the observations in each batch.

    Args:
        full_state_update (bool, optional): Whether to update the state with the
            new values.
            Defaults to ``True``.
        kwargs: Any keyword arguments.

    Examples:
        >>> from anomalib.metrics import MinMax
        >>> import torch
        ...
        >>> predictions = torch.tensor([0.0807, 0.6329, 0.0559, 0.9860, 0.3595])
        >>> minmax = MinMax()
        >>> minmax(predictions)
        (tensor(0.0559), tensor(0.9860))

        It is possible to update the minmax values with a new tensor of predictions.

        >>> new_predictions = torch.tensor([0.3251, 0.3169, 0.3072, 0.6247, 0.9999])
        >>> minmax.update(new_predictions)
        >>> minmax.compute()
        (tensor(0.0559), tensor(0.9999))
    """

    full_state_update: bool = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("min", torch.tensor(float("inf")), persistent=True)
        self.add_state("max", torch.tensor(float("-inf")), persistent=True)

        self.min = torch.tensor(float("inf"))
        self.max = torch.tensor(float("-inf"))

    def update(self, predictions: torch.Tensor, *args, **kwargs) -> None:
        """Update the min and max values."""
        del args, kwargs  # These variables are not used.

        self.max = torch.max(self.max, torch.max(predictions))
        self.min = torch.min(self.min, torch.min(predictions))

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return min and max values."""
        return self.min, self.max
