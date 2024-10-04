"""Base backend."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from pathlib import Path


class Backend(ABC):
    """Base backend."""

    @abstractmethod
    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        """Initialize the backend."""

    @abstractmethod
    def add_reference_images(self, image: str | Path) -> None:
        """Add reference images for k-shot."""

    @abstractmethod
    def predict(self, image: str | Path) -> str:
        """Predict the anomaly label."""
