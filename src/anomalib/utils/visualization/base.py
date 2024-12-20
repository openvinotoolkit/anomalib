"""Base visualization generator for anomaly detection.

This module provides the base visualization interface and common functionality used
across different visualization types. The key components include:

    - ``GeneratorResult``: Dataclass for standardized visualization outputs
    - ``VisualizationStep``: Enum for controlling when visualizations are generated
    - ``BaseVisualizer``: Abstract base class defining the visualization interface

Example:
    >>> from anomalib.utils.visualization import BaseVisualizer
    >>> # Create custom visualizer
    >>> class CustomVisualizer(BaseVisualizer):
    ...     def generate(self, **kwargs):
    ...         # Generate visualization
    ...         yield GeneratorResult(image=img)
    >>> # Use visualizer
    >>> vis = CustomVisualizer(visualize_on="batch")
    >>> results = vis.generate(image=input_img)

The module ensures consistent visualization behavior and output formats across
different visualization implementations.
"""

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
    """Base visualization generator.

    Deprecated: This class will be removed in v2.0.0 release.
    """

    def __init__(self, visualize_on: VisualizationStep) -> None:
        import warnings

        warnings.warn(
            "BaseVisualizer is deprecated and will be removed in v2.0.0 release.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.visualize_on = visualize_on

    @abstractmethod
    def generate(self, **kwargs) -> Iterator[GeneratorResult]:
        """Generate images and return them as an iterator."""
        raise NotImplementedError

    def __call__(self, **kwargs) -> Iterator[GeneratorResult]:
        """Call generate method."""
        return self.generate(**kwargs)
