"""Hugging Face backend for Vision Language Models (VLMs).

This module implements a backend for using Hugging Face models for vision-language
tasks in anomaly detection. The backend handles:

- Loading models and processors from Hugging Face Hub
- Processing images into model inputs
- Few-shot learning with reference images
- Model inference and response processing

Example:
    >>> from anomalib.models.image.vlm_ad.backends import Huggingface
    >>> backend = Huggingface(model_name="llava-hf/llava-1.5-7b-hf")  # doctest: +SKIP
    >>> backend.add_reference_images("normal_image.jpg")  # doctest: +SKIP
    >>> response = backend.predict("test.jpg", prompt)  # doctest: +SKIP

Args:
    model_name (str): Name of the Hugging Face model to use (e.g.
        ``"llava-hf/llava-1.5-7b-hf"``)

See Also:
    - :class:`Backend`: Base class for VLM backends
    - :class:`ChatGPT`: Alternative backend using OpenAI models
    - :class:`Ollama`: Alternative backend using Ollama models
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from lightning_utilities.core.imports import module_available
from PIL import Image

from anomalib.models.image.vlm_ad.utils import Prompt

from .base import Backend

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.processing_utils import ProcessorMixin

if module_available("transformers"):
    import transformers
else:
    transformers = None


logger = logging.getLogger(__name__)


class Huggingface(Backend):
    """Hugging Face backend for vision-language anomaly detection.

    This class implements a backend for using Hugging Face vision-language models for
    anomaly detection. It handles:

    - Loading models and processors from Hugging Face Hub
    - Processing images into model inputs
    - Few-shot learning with reference images
    - Model inference and response processing

    Args:
        model_name (str): Name of the Hugging Face model to use (e.g.
            ``"llava-hf/llava-1.5-7b-hf"``)

    Example:
        >>> from anomalib.models.image.vlm_ad.backends import Huggingface
        >>> backend = Huggingface(  # doctest: +SKIP
        ...     model_name="llava-hf/llava-1.5-7b-hf"
        ... )
        >>> backend.add_reference_images("normal_image.jpg")  # doctest: +SKIP
        >>> response = backend.predict("test.jpg", prompt)  # doctest: +SKIP

    Raises:
        ValueError: If transformers package is not installed

    See Also:
        - :class:`Backend`: Base class for VLM backends
        - :class:`ChatGPT`: Alternative backend using OpenAI models
        - :class:`Ollama`: Alternative backend using Ollama models
    """

    def __init__(
        self,
        model_name: str,
    ) -> None:
        """Initialize the Huggingface backend.

        Args:
            model_name (str): Name of the Hugging Face model to use
        """
        self.model_name: str = model_name
        self._ref_images: list[str] = []
        self._processor: ProcessorMixin | None = None
        self._model: PreTrainedModel | None = None

    @property
    def processor(self) -> "ProcessorMixin":
        """Get the Hugging Face processor.

        Returns:
            ProcessorMixin: Initialized processor for the model

        Raises:
            ValueError: If transformers package is not installed
        """
        if self._processor is None:
            if transformers is None:
                msg = "transformers is not installed."
                raise ValueError(msg)
            self._processor = transformers.LlavaNextProcessor.from_pretrained(self.model_name)
        return self._processor

    @property
    def model(self) -> "PreTrainedModel":
        """Get the Hugging Face model.

        Returns:
            PreTrainedModel: Initialized model instance

        Raises:
            ValueError: If transformers package is not installed
        """
        if self._model is None:
            if transformers is None:
                msg = "transformers is not installed."
                raise ValueError(msg)
            self._model = transformers.LlavaNextForConditionalGeneration.from_pretrained(self.model_name)
        return self._model

    @staticmethod
    def _generate_message(content: str, images: list[str] | None) -> dict:
        """Generate a message for the model.

        Args:
            content (str): Text content of the message
            images (list[str] | None): List of image paths to include in message

        Returns:
            dict: Formatted message dictionary with role and content
        """
        message: dict[str, str | list[dict]] = {"role": "user"}
        _content: list[dict[str, str]] = [{"type": "text", "text": content}]
        if images is not None:
            _content.extend([{"type": "image"} for _ in images])
        message["content"] = _content
        return message

    def add_reference_images(self, image: str | Path) -> None:
        """Add reference images for few-shot learning.

        Args:
            image (str | Path): Path to the reference image file
        """
        self._ref_images.append(Image.open(image))

    @property
    def num_reference_images(self) -> int:
        """Get the number of reference images.

        Returns:
            int: Number of reference images added
        """
        return len(self._ref_images)

    def predict(self, image_path: str | Path, prompt: Prompt) -> str:
        """Predict whether an image contains anomalies.

        Args:
            image_path (str | Path): Path to the image to analyze
            prompt (Prompt): Prompt object containing few-shot and prediction prompts

        Returns:
            str: Model's prediction response
        """
        image = Image.open(image_path)
        messages: list[dict] = []

        if len(self._ref_images) > 0:
            messages.append(self._generate_message(content=prompt.few_shot, images=self._ref_images))

        messages.append(self._generate_message(content=prompt.predict, images=[image]))
        processed_prompt = [self.processor.apply_chat_template(messages, add_generation_prompt=True)]

        images = [*self._ref_images, image]
        inputs = self.processor(images, processed_prompt, return_tensors="pt", padding=True).to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        return self.processor.decode(outputs[0], skip_special_tokens=True)
