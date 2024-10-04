"""Huggingface backend."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from enum import Enum
from pathlib import Path

from PIL import Image
from transformers.modeling_utils import PreTrainedModel

from anomalib.utils.exceptions import try_import

from .base import Backend
from .dataclasses import Prompt

if try_import("transformers"):
    import transformers
    from transformers.modeling_utils import PreTrainedModel
    from transformers.processing_utils import ProcessorMixin
else:
    transformers = None


logger = logging.getLogger(__name__)


class LlavaNextModels(Enum):
    """Available models."""

    VICUNA_7B = "llava-hf/llava-v1.6-vicuna-7b-hf"
    VICUNA_13B = "llava-hf/llava-v1.6-vicuna-13b-hf"
    MISTRAL_7B = "llava-hf/llava-v1.6-mistral-7b-hf"


class Huggingface(Backend):
    """Huggingface backend."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | LlavaNextModels = LlavaNextModels.VICUNA_7B,
    ) -> None:
        """Initialize the Huggingface backend."""
        if api_key:
            logger.warning("API key is not required for Huggingface backend.")
        self.model_name: str = LlavaNextModels(model_name).value
        self._ref_images: list[str] = []
        self._processor: ProcessorMixin | None = None
        self._model: PreTrainedModel | None = None

    @property
    def prompt(self) -> Prompt:
        """Get the Ollama prompt."""
        return Prompt(
            predict=(
                "You are given an image. It is either normal or anomalous."
                " First say 'YES' if the image is anomalous, or 'NO' if it is normal.\n"
                "Then give the reason for your decision.\n"
                "For example, 'YES: The image has a crack on the wall.'"
            ),
            few_shot=(
                "These are a few examples of normal picture without any anomalies."
                " You have to use these to determine if the image I provide in the next"
                " chat is normal or anomalous."
            ),
        )

    @property
    def processor(self) -> ProcessorMixin:
        if self._processor is None:
            if transformers is None:
                msg = "transformers is not installed."
                raise ValueError(msg)
            self._processor = transformers.LlavaNextProcessor.from_pretrained(self.model_name)
        return self._processor

    @property
    def model(self) -> PreTrainedModel:
        if self._model is None:
            if transformers is None:
                msg = "transformers is not installed."
                raise ValueError(msg)
            self._model: PreTrainedModel = transformers.LlavaNextForConditionalGeneration.from_pretrained(
                self.model_name,
            )
        return self._model

    @staticmethod
    def _generate_message(content: str, images: list[str] | None) -> dict:
        """Generate a message."""
        message = {"role": "user"}
        message["content"] = [{"type": "text", "text": content}]
        if images is not None:
            for _ in images:
                message["content"].append({"type": "image"})
        return message

    def add_reference_images(self, image: str | Path) -> None:
        self._ref_images.append(Image.open(image))

    def predict(self, image_path: str | Path) -> str:
        """Predict the anomaly label."""
        image = Image.open(image_path)
        messages = []

        if len(self._ref_images) > 0:
            messages.append(self._generate_message(content=self.prompt.few_shot, images=self._ref_images))

        messages.append(self._generate_message(content=self.prompt.predict, images=[image]))
        prompt = [self.processor.apply_chat_template(messages, add_generation_prompt=True)]

        images = [*self._ref_images, image]
        inputs = self.processor(images, prompt, return_tensors="pt", padding=True).to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        result = self.processor.decode(outputs[0], skip_special_tokens=True)
        print(result)
        return result
