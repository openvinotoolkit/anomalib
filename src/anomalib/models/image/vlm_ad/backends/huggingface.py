"""Huggingface backend."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from lightning_utilities.core.imports import package_available
from PIL import Image

from anomalib.models.image.vlm_ad.utils import Prompt

from .base import Backend

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.processing_utils import ProcessorMixin

if package_available("transformers"):
    import transformers
else:
    transformers = None


logger = logging.getLogger(__name__)


class Huggingface(Backend):
    """Huggingface backend."""

    def __init__(
        self,
        model_name: str,
    ) -> None:
        """Initialize the Huggingface backend."""
        self.model_name: str = model_name
        self._ref_images: list[str] = []
        self._processor: ProcessorMixin | None = None
        self._model: PreTrainedModel | None = None

    @property
    def processor(self) -> "ProcessorMixin":
        """Get the Huggingface processor."""
        if self._processor is None:
            if transformers is None:
                msg = "transformers is not installed."
                raise ValueError(msg)
            self._processor = transformers.LlavaNextProcessor.from_pretrained(self.model_name)
        return self._processor

    @property
    def model(self) -> "PreTrainedModel":
        """Get the Huggingface model."""
        if self._model is None:
            if transformers is None:
                msg = "transformers is not installed."
                raise ValueError(msg)
            self._model = transformers.LlavaNextForConditionalGeneration.from_pretrained(self.model_name)
        return self._model

    @staticmethod
    def _generate_message(content: str, images: list[str] | None) -> dict:
        """Generate a message."""
        message: dict[str, str | list[dict]] = {"role": "user"}
        _content: list[dict[str, str]] = [{"type": "text", "text": content}]
        if images is not None:
            _content.extend([{"type": "image"} for _ in images])
        message["content"] = _content
        return message

    def add_reference_images(self, image: str | Path) -> None:
        """Add reference images for k-shot."""
        self._ref_images.append(Image.open(image))

    @property
    def num_reference_images(self) -> int:
        """Get the number of reference images."""
        return len(self._ref_images)

    def predict(self, image_path: str | Path, prompt: Prompt) -> str:
        """Predict the anomaly label."""
        image = Image.open(image_path)
        messages: list[dict] = []

        if len(self._ref_images) > 0:
            messages.append(self._generate_message(content=prompt.few_shot, images=self._ref_images))

        messages.append(self._generate_message(content=prompt.predict, images=[image]))
        processed_prompt = [self.processor.apply_chat_template(messages, add_generation_prompt=True)]

        images = [*self._ref_images, image]
        inputs = self.processor(images, processed_prompt, return_tensors="pt", padding=True).to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        result = self.processor.decode(outputs[0], skip_special_tokens=True)
        print(result)
        return result
