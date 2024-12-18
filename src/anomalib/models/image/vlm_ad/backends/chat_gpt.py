"""ChatGPT backend for Vision Language Models (VLMs).

This module implements a backend for using OpenAI's ChatGPT model for vision-language
tasks in anomaly detection. The backend handles:

- Authentication with OpenAI API
- Encoding and sending images
- Prompting the model
- Processing responses

Example:
    >>> from anomalib.models.image.vlm_ad.backends import ChatGPT
    >>> backend = ChatGPT(model_name="gpt-4-vision-preview")  # doctest: +SKIP
    >>> backend.add_reference_images("normal_image.jpg")  # doctest: +SKIP
    >>> response = backend.predict("test.jpg", prompt)  # doctest: +SKIP

Args:
    model_name (str): Name of the ChatGPT model to use (e.g. ``"gpt-4-vision-preview"``)
    api_key (str | None, optional): OpenAI API key. If not provided, will attempt to
        load from environment. Defaults to ``None``.

See Also:
    - :class:`Backend`: Base class for VLM backends
    - :class:`Huggingface`: Alternative backend using Hugging Face models
    - :class:`Ollama`: Alternative backend using Ollama models
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import base64
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from lightning_utilities.core.imports import module_available

from anomalib.models.image.vlm_ad.utils import Prompt

from .base import Backend

if module_available("openai"):
    from openai import OpenAI
else:
    OpenAI = None

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)


class ChatGPT(Backend):
    """OpenAI ChatGPT backend for vision-language anomaly detection.

    This class implements a backend for using OpenAI's ChatGPT models with vision
    capabilities (e.g. GPT-4V) for anomaly detection. It handles:

    - Authentication with OpenAI API
    - Image encoding and formatting
    - Few-shot learning with reference images
    - Model prompting and response processing

    Args:
        model_name (str): Name of the ChatGPT model to use (e.g.
            ``"gpt-4-vision-preview"``)
        api_key (str | None, optional): OpenAI API key. If not provided, will
            attempt to load from environment. Defaults to ``None``.

    Example:
        >>> from anomalib.models.image.vlm_ad.backends import ChatGPT
        >>> backend = ChatGPT(model_name="gpt-4-vision-preview")  # doctest: +SKIP
        >>> backend.add_reference_images("normal_image.jpg")  # doctest: +SKIP
        >>> response = backend.predict("test.jpg", prompt)  # doctest: +SKIP

    Raises:
        ImportError: If OpenAI package is not installed
        ValueError: If no API key is provided or found in environment

    See Also:
        - :class:`Backend`: Base class for VLM backends
        - :class:`Huggingface`: Alternative backend using Hugging Face models
        - :class:`Ollama`: Alternative backend using Ollama models
    """

    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        """Initialize the ChatGPT backend."""
        self._ref_images_encoded: list[str] = []
        self.model_name: str = model_name
        self._client: OpenAI | None = None
        self.api_key = self._get_api_key(api_key)

    @property
    def client(self) -> OpenAI:
        """Get the OpenAI client.

        Returns:
            OpenAI: Initialized OpenAI client instance

        Raises:
            ImportError: If OpenAI package is not installed
        """
        if OpenAI is None:
            msg = "OpenAI is not installed. Please install it to use ChatGPT backend."
            raise ImportError(msg)
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def add_reference_images(self, image: str | Path) -> None:
        """Add reference images for few-shot learning.

        Args:
            image (str | Path): Path to the reference image file
        """
        self._ref_images_encoded.append(self._encode_image_to_url(image))

    @property
    def num_reference_images(self) -> int:
        """Get the number of reference images.

        Returns:
            int: Number of reference images added for few-shot learning
        """
        return len(self._ref_images_encoded)

    def predict(self, image: str | Path, prompt: Prompt) -> str:
        """Predict whether an image contains anomalies.

        Args:
            image (str | Path): Path to the image file to analyze
            prompt (Prompt): Prompt object containing few-shot and prediction
                prompts

        Returns:
            str: Model's response indicating if anomalies were detected
        """
        image_encoded = self._encode_image_to_url(image)
        messages = []

        # few-shot
        if len(self._ref_images_encoded) > 0:
            messages.append(self._generate_message(content=prompt.few_shot, images=self._ref_images_encoded))

        messages.append(self._generate_message(content=prompt.predict, images=[image_encoded]))

        response: ChatCompletion = self.client.chat.completions.create(messages=messages, model=self.model_name)
        return response.choices[0].message.content

    @staticmethod
    def _generate_message(content: str, images: list[str] | None) -> dict:
        """Generate a message for the ChatGPT API.

        Args:
            content (str): Text content of the message
            images (list[str] | None): List of base64-encoded image URLs

        Returns:
            dict: Formatted message dictionary for the API
        """
        message: dict[str, list[dict] | str] = {"role": "user"}
        if images is not None:
            _content: list[dict[str, str | dict]] = [{"type": "text", "text": content}]
            _content.extend([{"type": "image_url", "image_url": {"url": image}} for image in images])
            message["content"] = _content
        else:
            message["content"] = content
        return message

    def _encode_image_to_url(self, image: str | Path) -> str:
        """Encode an image file to a base64 URL string.

        Args:
            image (str | Path): Path to the image file

        Returns:
            str: Base64-encoded image URL string
        """
        image_path = Path(image)
        extension = image_path.suffix
        base64_encoded = self._encode_image_to_base_64(image_path)
        return f"data:image/{extension};base64,{base64_encoded}"

    @staticmethod
    def _encode_image_to_base_64(image: str | Path) -> str:
        """Encode an image file to base64.

        Args:
            image (str | Path): Path to the image file

        Returns:
            str: Base64-encoded image string
        """
        image = Path(image)
        return base64.b64encode(image.read_bytes()).decode("utf-8")

    def _get_api_key(self, api_key: str | None = None) -> str:
        """Get the OpenAI API key.

        Attempts to get the API key in the following order:
        1. From the provided argument
        2. From environment variable ``OPENAI_API_KEY``
        3. From ``.env`` file

        Args:
            api_key (str | None, optional): API key provided directly. Defaults to
                ``None``.

        Returns:
            str: Valid OpenAI API key

        Raises:
            ValueError: If no API key is found
        """
        if api_key is None:
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            msg = (
                f"OpenAI API key must be provided to use {self.model_name}."
                " Please provide the API key in the constructor, or set the OPENAI_API_KEY environment variable"
                " or in a `.env` file."
            )
            raise ValueError(msg)
        return api_key
