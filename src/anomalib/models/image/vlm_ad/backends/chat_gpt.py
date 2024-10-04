"""ChatGPT backend."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import base64
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from anomalib.utils.exceptions import try_import

from .base import Backend
from .dataclasses import Prompt

if try_import("openai"):
    from openai import OpenAI
else:
    OpenAI = None

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)


class ChatGPT(Backend):
    """ChatGPT backend."""

    def __init__(self, api_key: str | None = None, model_name: str = "gpt-4o-mini") -> None:
        """Initialize the ChatGPT backend."""
        if api_key is None:
            msg = "API key is required for ChatGPT backend."
            raise ValueError(msg)
        self.api_key = api_key
        self._ref_images_encoded: list[str] = []
        self.model_name: str = model_name
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        """Get the OpenAI client."""
        if OpenAI is None:
            msg = "OpenAI is not installed. Please install it to use ChatGPT backend."
            raise ImportError(msg)
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def add_reference_images(self, image: str | Path) -> None:
        """Add reference images for k-shot."""
        self._ref_images_encoded.append(self._encode_image_to_url(image))

    def predict(self, image: str | Path) -> str:
        """Predict the anomaly label."""
        image_encoded = self._encode_image_to_url(image)
        messages = []

        # few-shot
        if len(self._ref_images_encoded) > 0:
            messages.append(self._generate_message(content=self.prompt.few_shot, images=self._ref_images_encoded))

        messages.append(self._generate_message(content=self.prompt.predict, images=[image_encoded]))

        response: ChatCompletion = self.client.chat.completions.create(messages=messages, model=self.model_name)
        return response.choices[0].message.content

    @staticmethod
    def _generate_message(content: str, images: list[str] | None) -> dict:
        """Generate a message."""
        message = {"role": "user"}
        if images is None:
            message["content"] = content
        else:
            message["content"] = [{"type": "text", "text": content}]
            for image in images:
                message["content"].append({"type": "image_url", "image_url": {"url": image}})
        return message

    def _encode_image_to_url(self, image: str | Path) -> str:
        """Encode the image to base64 and embed in url string."""
        image_path = Path(image)
        extension = image_path.suffix
        base64_encoded = self._encode_image_to_base_64(image_path)
        return f"data:image/{extension};base64,{base64_encoded}"

    @staticmethod
    def _encode_image_to_base_64(image: str | Path) -> str:
        """Encode the image to base64."""
        image = Path(image)
        return base64.b64encode(image.read_bytes()).decode("utf-8")

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
