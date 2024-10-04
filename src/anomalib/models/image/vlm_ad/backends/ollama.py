"""Ollama backend.

Assumes that the Ollama service is running in the background.
See: https://github.com/ollama/ollama
Ensure that ollama is running. On linux: `ollama serve`
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

from anomalib.utils.exceptions import try_import

from .base import Backend
from .dataclasses import Prompt

if try_import("ollama"):
    from ollama import chat
    from ollama._client import _encode_image
else:
    chat = None

logger = logging.getLogger(__name__)


class Ollama(Backend):
    """Ollama backend."""

    def __init__(self, api_key: str | None = None, model_name: str = "llava") -> None:
        """Initialize the Ollama backend."""
        if api_key:
            logger.warning("API key is not required for Ollama backend.")
        self.model_name: str = model_name
        self._ref_images_encoded: list[str] = []

    def add_reference_images(self, image: str | Path) -> None:
        """Encode the image to base64."""
        self._ref_images_encoded.append(_encode_image(image))

    @property
    def prompt(self) -> Prompt:
        """Get the Ollama prompt."""
        return Prompt(
            predict=(
                "You are given an image. It is either normal or anomalous."
                "First say 'YES' if the image is anomalous, or 'NO' if it is normal.\n"
                "Then give the reason for your decision.\n"
                "For example, 'YES: The image has a crack on the wall.'"
            ),
            few_shot=(
                "These are a few examples of normal picture without any anomalies."
                " You have to use these to determine if the image I provide in the next"
                " chat is normal or anomalous."
            ),
        )

    @staticmethod
    def _generate_message(content: str, images: list[str] | None) -> dict:
        """Generate a message."""
        message = {"role": "user", "content": content}
        if images:
            message["images"] = images
        return message

    def predict(self, image: str | Path) -> str:
        """Predict the anomaly label."""
        if not chat:
            msg = "Ollama is not installed. Please install it using `pip install ollama`."
            raise ImportError(msg)
        image_encoded = _encode_image(image)
        messages = []

        # few-shot
        if len(self._ref_images_encoded) > 0:
            messages.append(
                self._generate_message(
                    content=self.prompt.few_shot,
                    images=self._ref_images_encoded,
                ),
            )

        messages.append(
            self._generate_message(content=self.prompt.predict, images=[image_encoded]),
        )

        response = chat(
            model=self.model_name,
            messages=messages,
        )
        return response["message"]["content"].strip()
