"""Base logger for image logging consistency across all loggers used in anomalib.

This module provides a base class that defines a common interface for logging images
across different logging backends used in anomalib.

Example:
    Create a custom image logger:

    >>> class CustomImageLogger(ImageLoggerBase):
    ...     def add_image(self, image, name=None):
    ...         # Custom implementation
    ...         pass

    Use the logger:

    >>> logger = CustomImageLogger()
    >>> logger.add_image(image_array, name="test_image")  # doctest: +SKIP
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod

import numpy as np
from matplotlib.figure import Figure


class ImageLoggerBase:
    """Base class that provides a common interface for logging images.

    This abstract base class ensures consistent image logging functionality across
    different logger implementations in anomalib.

    All custom image loggers should inherit from this class and implement the
    ``add_image`` method.
    """

    @abstractmethod
    def add_image(self, image: np.ndarray | Figure, name: str | None = None, **kwargs) -> None:
        """Log an image using the respective logger implementation.

        Args:
            image: Image to be logged, can be either a numpy array or matplotlib
                Figure
            name: Name/title of the image. Defaults to ``None``
            **kwargs: Additional keyword arguments passed to the specific logger
                implementation

        Raises:
            NotImplementedError: This is an abstract method that must be
                implemented by subclasses
        """
        raise NotImplementedError
