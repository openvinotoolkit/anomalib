"""Vision Language Model (VLM) backends for anomaly detection.

This module provides backend implementations for different Vision Language Models
(VLMs) that can be used for anomaly detection. The backends include:

- :class:`ChatGPT`: OpenAI's ChatGPT model
- :class:`Huggingface`: Models from Hugging Face Hub
- :class:`Ollama`: Open source LLM models via Ollama

Example:
    >>> from anomalib.models.image.vlm_ad.backends import ChatGPT
    >>> backend = ChatGPT()  # doctest: +SKIP
    >>> response = backend.generate(prompt="Describe this image")  # doctest: +SKIP

See Also:
    - :class:`Backend`: Base class for VLM backends
    - :class:`ChatGPT`: ChatGPT backend implementation
    - :class:`Huggingface`: Hugging Face backend implementation
    - :class:`Ollama`: Ollama backend implementation
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .base import Backend
from .chat_gpt import ChatGPT
from .huggingface import Huggingface
from .ollama import Ollama

__all__ = ["Backend", "ChatGPT", "Huggingface", "Ollama"]
