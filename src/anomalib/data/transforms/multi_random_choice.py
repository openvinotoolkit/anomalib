"""Multi random choice transform.

This transform randomly applies multiple transforms from a list of transforms.

Example:
    >>> import torchvision.transforms.v2 as v2
    >>> transforms = [
    ...     v2.RandomHorizontalFlip(p=1.0),
    ...     v2.ColorJitter(brightness=0.5),
    ...     v2.RandomRotation(10),
    ... ]
    >>> # Apply 1-2 random transforms with equal probability
    >>> transform = MultiRandomChoice(transforms, num_transforms=2)
    >>> # Always apply exactly 2 transforms with custom probabilities
    >>> transform = MultiRandomChoice(
    ...     transforms,
    ...     probabilities=[0.5, 0.3, 0.2],
    ...     num_transforms=2,
    ...     fixed_num_transforms=True
    ... )
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Sequence

import torch
from torchvision.transforms import v2


class MultiRandomChoice(v2.Transform):
    """Apply multiple transforms randomly picked from a list.

    This transform does not support torchscript.

    Args:
        transforms: List of transformations to choose from.
        probabilities: Probability of each transform being picked. If ``None``
            (default), all transforms have equal probability. If provided,
            probabilities will be normalized to sum to 1.
        num_transforms: Maximum number of transforms to apply at once.
            Defaults to ``1``.
        fixed_num_transforms: If ``True``, always applies exactly
            ``num_transforms`` transforms. If ``False``, randomly picks between
            1 and ``num_transforms``. Defaults to ``False``.

    Raises:
        TypeError: If ``transforms`` is not a sequence of callables.
        ValueError: If length of ``probabilities`` does not match length of
            ``transforms``.

    Example:
        >>> import torchvision.transforms.v2 as v2
        >>> transforms = [
        ...     v2.RandomHorizontalFlip(p=1.0),
        ...     v2.ColorJitter(brightness=0.5),
        ...     v2.RandomRotation(10),
        ... ]
        >>> # Apply 1-2 random transforms with equal probability
        >>> transform = MultiRandomChoice(transforms, num_transforms=2)
        >>> # Always apply exactly 2 transforms with custom probabilities
        >>> transform = MultiRandomChoice(
        ...     transforms,
        ...     probabilities=[0.5, 0.3, 0.2],
        ...     num_transforms=2,
        ...     fixed_num_transforms=True
        ... )
    """

    def __init__(
        self,
        transforms: Sequence[Callable],
        probabilities: list[float] | None = None,
        num_transforms: int = 1,
        fixed_num_transforms: bool = False,
    ) -> None:
        if not isinstance(transforms, Sequence):
            msg = "Argument transforms should be a sequence of callables"
            raise TypeError(msg)

        if probabilities is None:
            probabilities = [1.0] * len(transforms)
        elif len(probabilities) != len(transforms):
            msg = f"Length of p doesn't match the number of transforms: {len(probabilities)} != {len(transforms)}"
            raise ValueError(msg)

        super().__init__()

        self.transforms = transforms
        total = sum(probabilities)
        self.probabilities = [probability / total for probability in probabilities]

        self.num_transforms = num_transforms
        self.fixed_num_transforms = fixed_num_transforms

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Apply randomly selected transforms to the input.

        Args:
            *inputs: Input tensors to transform.

        Returns:
            Transformed tensor(s).
        """
        # First determine number of transforms to apply
        num_transforms = (
            self.num_transforms if self.fixed_num_transforms else int(torch.randint(self.num_transforms, (1,)) + 1)
        )
        # Get transforms
        idx = torch.multinomial(torch.tensor(self.probabilities), num_transforms).tolist()
        transform = v2.Compose([self.transforms[i] for i in idx])
        return transform(*inputs)
