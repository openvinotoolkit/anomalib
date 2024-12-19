"""Base backend for Vision Language Models (VLMs).

This module provides the abstract base class for VLM backends used in anomaly detection.
The backends handle communication with different VLM services and models.

Example:
    >>> from anomalib.models.image.vlm_ad.backends import Backend
    >>> class CustomBackend(Backend):
    ...     def __init__(self, model_name: str) -> None:
    ...         super().__init__(model_name)
    ...     def add_reference_images(self, image: str) -> None:
    ...         pass
    ...     def predict(self, image: str, prompt: Prompt) -> str:
    ...         return "normal"
    ...     @property
    ...     def num_reference_images(self) -> int:
    ...         return 0

See Also:
    - :class:`ChatGPT`: OpenAI's ChatGPT backend implementation
    - :class:`Huggingface`: Hugging Face models backend implementation
    - :class:`Ollama`: Ollama models backend implementation
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from pathlib import Path

from anomalib.models.image.vlm_ad.utils import Prompt


class Backend(ABC):
    """Abstract base class for Vision Language Model (VLM) backends.

    This class defines the interface that all VLM backends must implement. Backends
    handle communication with different VLM services and models for anomaly detection.

    Example:
        >>> from anomalib.models.image.vlm_ad.backends import Backend
        >>> class CustomBackend(Backend):
        ...     def __init__(self, model_name: str) -> None:
        ...         super().__init__(model_name)
        ...     def add_reference_images(self, image: str) -> None:
        ...         pass
        ...     def predict(self, image: str, prompt: Prompt) -> str:
        ...         return "normal"
        ...     @property
        ...     def num_reference_images(self) -> int:
        ...         return 0

    See Also:
        - :class:`ChatGPT`: OpenAI's ChatGPT backend implementation
        - :class:`Huggingface`: Hugging Face models backend implementation
        - :class:`Ollama`: Ollama models backend implementation
    """

    @abstractmethod
    def __init__(self, model_name: str) -> None:
        """Initialize the VLM backend.

        Args:
            model_name (str): Name or identifier of the VLM model to use
        """

    @abstractmethod
    def add_reference_images(self, image: str | Path) -> None:
        """Add reference images for few-shot learning.

        The backend stores these images to use as examples when making predictions.

        Args:
            image (str | Path): Path to the reference image file
        """

    @abstractmethod
    def predict(self, image: str | Path, prompt: Prompt) -> str:
        """Predict whether an image contains anomalies.

        Args:
            image (str | Path): Path to the image file to analyze
            prompt (Prompt): Prompt template to use for querying the VLM

        Returns:
            str: Prediction result from the VLM
        """

    @property
    @abstractmethod
    def num_reference_images(self) -> int:
        """Get the number of stored reference images.

        Returns:
            int: Count of reference images currently stored in the backend
        """
