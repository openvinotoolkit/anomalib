"""Base logger for image logging consistency across all loggers used in anomalib."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod

import numpy as np
from matplotlib.figure import Figure


class ImageLoggerBase:
    """Adds a common interface for logging the images."""

    @abstractmethod
    def add_image(self, image: np.ndarray | Figure, name: str | None = None, **kwargs) -> None:
        """Interface to log images in the respective loggers."""
        raise NotImplementedError
