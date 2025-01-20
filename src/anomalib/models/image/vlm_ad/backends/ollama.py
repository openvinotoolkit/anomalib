"""Ollama backend for Vision Language Models (VLMs).

This module implements a backend for using Ollama models for vision-language tasks in
anomaly detection. The backend handles:

- Communication with local Ollama service
- Image encoding and formatting
- Few-shot learning with reference images
- Model inference and response processing

Example:
    >>> from anomalib.models.image.vlm_ad.backends import Ollama
    >>> backend = Ollama(model_name="llava")  # doctest: +SKIP
    >>> backend.add_reference_images("normal_image.jpg")  # doctest: +SKIP
    >>> response = backend.predict("test.jpg", prompt)  # doctest: +SKIP

Note:
    Requires Ollama service to be running in the background:

    - Linux: Run ``ollama serve``
    - Mac/Windows: Launch Ollama application from applications list

    See `Ollama documentation <https://github.com/ollama/ollama>`_ for setup details.

Args:
    model_name (str): Name of the Ollama model to use (e.g. ``"llava"``)

See Also:
    - :class:`Backend`: Base class for VLM backends
    - :class:`ChatGPT`: Alternative backend using OpenAI models
    - :class:`Huggingface`: Alternative backend using Hugging Face models
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

from lightning_utilities.core.imports import module_available

from anomalib.models.image.vlm_ad.utils import Prompt

from .base import Backend

if module_available("ollama"):
    from ollama import Image, chat
else:
    chat = None

logger = logging.getLogger(__name__)


class Ollama(Backend):
    """Ollama backend for vision-language anomaly detection.

    This class implements a backend for using Ollama models with vision capabilities
    for anomaly detection. It handles:

    - Communication with local Ollama service
    - Image encoding and formatting
    - Few-shot learning with reference images
    - Model inference and response processing

    Args:
        model_name (str): Name of the Ollama model to use (e.g. ``"llava"``)

    Example:
        >>> from anomalib.models.image.vlm_ad.backends import Ollama
        >>> backend = Ollama(model_name="llava")  # doctest: +SKIP
        >>> backend.add_reference_images("normal_image.jpg")  # doctest: +SKIP
        >>> response = backend.predict("test.jpg", prompt)  # doctest: +SKIP

    Note:
        Requires Ollama service to be running in the background:

        - Linux: Run ``ollama serve``
        - Mac/Windows: Launch Ollama application from applications list

    See Also:
        - :class:`Backend`: Base class for VLM backends
        - :class:`ChatGPT`: Alternative backend using OpenAI models
        - :class:`Huggingface`: Alternative backend using Hugging Face models
    """

    def __init__(self, model_name: str) -> None:
        """Initialize the Ollama backend.

        Args:
            model_name (str): Name of the Ollama model to use
        """
        self.model_name: str = model_name
        self._ref_images_encoded: list[str] = []

    def add_reference_images(self, image: str | Path) -> None:
        """Add and encode reference images for few-shot learning.

        The images are encoded to base64 format for sending to the Ollama service.

        Args:
            image (str | Path): Path to the reference image file
        """
        self._ref_images_encoded.append(Image(value=image))

    @property
    def num_reference_images(self) -> int:
        """Get the number of reference images.

        Returns:
            int: Number of reference images added
        """
        return len(self._ref_images_encoded)

    @staticmethod
    def _generate_message(content: str, images: list[str] | None) -> dict:
        """Generate a message for the Ollama chat API.

        Args:
            content (str): Text content of the message
            images (list[str] | None): List of base64 encoded images to include

        Returns:
            dict: Formatted message dictionary with role, content and optional images
        """
        message: dict[str, str | list[str]] = {"role": "user", "content": content}
        if images:
            message["images"] = images
        return message

    def predict(self, image: str | Path, prompt: Prompt) -> str:
        """Predict whether an image contains anomalies.

        Args:
            image (str | Path): Path to the image to analyze
            prompt (Prompt): Prompt object containing few-shot and prediction prompts

        Returns:
            str: Model's prediction response

        Raises:
            ImportError: If Ollama package is not installed
        """
        if not chat:
            msg = "Ollama is not installed. Please install it using `pip install ollama`."
            raise ImportError(msg)
        image_encoded = Image(value=image)
        messages = []

        # few-shot
        if len(self._ref_images_encoded) > 0:
            messages.append(self._generate_message(content=prompt.few_shot, images=self._ref_images_encoded))

        messages.append(self._generate_message(content=prompt.predict, images=[image_encoded]))

        response = chat(
            model=self.model_name,
            messages=messages,
        )
        return response["message"]["content"].strip()
