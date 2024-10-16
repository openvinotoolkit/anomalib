"""ChatGPT backend."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import base64
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from lightning_utilities.core.imports import package_available

from anomalib.models.image.vlm_ad.utils import Prompt

from .base import Backend

if package_available("openai"):
    from openai import OpenAI
else:
    OpenAI = None

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)


class ChatGPT(Backend):
    """ChatGPT backend."""

    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        """Initialize the ChatGPT backend."""
        self._ref_images_encoded: list[str] = []
        self.model_name: str = model_name
        self._client: OpenAI | None = None
        self.api_key = self._get_api_key(api_key)

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

    @property
    def num_reference_images(self) -> int:
        """Get the number of reference images."""
        return len(self._ref_images_encoded)

    def predict(self, image: str | Path, prompt: Prompt) -> str:
        """Predict the anomaly label."""
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
        """Generate a message."""
        message: dict[str, list[dict] | str] = {"role": "user"}
        if images is not None:
            _content: list[dict[str, str | dict]] = [{"type": "text", "text": content}]
            _content.extend([{"type": "image_url", "image_url": {"url": image}} for image in images])
            message["content"] = _content
        else:
            message["content"] = content
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

    def _get_api_key(self, api_key: str | None = None) -> str:
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
