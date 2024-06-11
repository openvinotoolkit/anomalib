"""Base visualization generator."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np


@dataclass
class GeneratorResult:
    """Generator result.

    All visualization generators are expected to return this object.
    It is to ensure that the result is consistent across all generators.
    """

    image: np.ndarray
    file_name: str | Path | None = None


class VisualizationStep(str, Enum):
    """Identify step on which to generate images."""

    BATCH = "batch"
    STAGE_END = "stage_end"


class BaseVisualizer(ABC):
    """Base visualization generator."""

    def __init__(self, visualize_on: VisualizationStep) -> None:
        self.visualize_on = visualize_on

    @abstractmethod
    def generate(self, **kwargs) -> Iterator[GeneratorResult]:
        """Generate images and return them as an iterator."""
        raise NotImplementedError

    def __call__(self, **kwargs) -> Iterator[GeneratorResult]:
        """Call generate method."""
        return self.generate(**kwargs)
