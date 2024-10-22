"""Base backend."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from pathlib import Path

from anomalib.models.image.vlm_ad.utils import Prompt


class Backend(ABC):
    """Base backend."""

    @abstractmethod
    def __init__(self, model_name: str) -> None:
        """Initialize the backend."""

    @abstractmethod
    def add_reference_images(self, image: str | Path) -> None:
        """Add reference images for k-shot."""

    @abstractmethod
    def predict(self, image: str | Path, prompt: Prompt) -> str:
        """Predict the anomaly label."""

    @property
    @abstractmethod
    def num_reference_images(self) -> int:
        """Get the number of reference images."""
