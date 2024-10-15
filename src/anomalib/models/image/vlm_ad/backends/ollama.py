"""Ollama backend.

Assumes that the Ollama service is running in the background.
See: https://github.com/ollama/ollama
Ensure that ollama is running. On linux: `ollama serve`
On Mac and Windows ensure that the ollama service is running by launching from the application list.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

from lightning_utilities.core.imports import package_available

from anomalib.models.image.vlm_ad.utils import Prompt

from .base import Backend

if package_available("ollama"):
    from ollama import chat
    from ollama._client import _encode_image
else:
    chat = None

logger = logging.getLogger(__name__)


class Ollama(Backend):
    """Ollama backend."""

    def __init__(self, model_name: str) -> None:
        """Initialize the Ollama backend."""
        self.model_name: str = model_name
        self._ref_images_encoded: list[str] = []

    def add_reference_images(self, image: str | Path) -> None:
        """Encode the image to base64."""
        self._ref_images_encoded.append(_encode_image(image))

    @property
    def num_reference_images(self) -> int:
        """Get the number of reference images."""
        return len(self._ref_images_encoded)

    @staticmethod
    def _generate_message(content: str, images: list[str] | None) -> dict:
        """Generate a message."""
        message: dict[str, str | list[str]] = {"role": "user", "content": content}
        if images:
            message["images"] = images
        return message

    def predict(self, image: str | Path, prompt: Prompt) -> str:
        """Predict the anomaly label."""
        if not chat:
            msg = "Ollama is not installed. Please install it using `pip install ollama`."
            raise ImportError(msg)
        image_encoded = _encode_image(image)
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
