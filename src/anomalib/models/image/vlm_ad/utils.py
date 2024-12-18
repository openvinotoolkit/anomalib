"""Utility classes and functions for Vision Language Model (VLM) based anomaly detection.

This module provides utility classes for VLM-based anomaly detection:

- :class:`Prompt`: Dataclass for storing few-shot and prediction prompts
- :class:`ModelName`: Enum of supported VLM models

Example:
    >>> from anomalib.models.image.vlm_ad.utils import Prompt, ModelName
    >>> prompt = Prompt(  # doctest: +SKIP
    ...     few_shot="These are normal examples...",
    ...     predict="Is this image normal or anomalous?"
    ... )
    >>> model_name = ModelName.LLAMA_OLLAMA  # doctest: +SKIP

See Also:
    - :class:`VlmAd`: Main model class using these utilities
    - :mod:`.backends`: VLM backend implementations using these utilities
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum


@dataclass
class Prompt:
    """Dataclass for storing prompts used in VLM-based anomaly detection.

    This class stores two types of prompts used when querying vision language models:

    - Few-shot prompt: Used to provide context about normal examples
    - Prediction prompt: Used to query about a specific test image

    Args:
        few_shot (str): Prompt template for few-shot learning with reference normal
            images. Used to establish context about what constitutes normal.
        predict (str): Prompt template for querying about test images. Used to ask
            the model whether a given image contains anomalies.

    Example:
        >>> from anomalib.models.image.vlm_ad.utils import Prompt
        >>> prompt = Prompt(  # doctest: +SKIP
        ...     few_shot="Here are some examples of normal items...",
        ...     predict="Is this image normal or does it contain defects?"
        ... )

    See Also:
        - :class:`VlmAd`: Main model class using these prompts
        - :mod:`.backends`: VLM backend implementations using these prompts
    """

    few_shot: str
    predict: str


class ModelName(Enum):
    """Enumeration of supported Vision Language Models (VLMs).

    This enum defines the available VLM models that can be used for anomaly detection:

    - ``LLAMA_OLLAMA``: LLaVA model running via Ollama
    - ``GPT_4O_MINI``: GPT-4O Mini model
    - ``VICUNA_7B_HF``: LLaVA v1.6 with Vicuna 7B base from Hugging Face
    - ``VICUNA_13B_HF``: LLaVA v1.6 with Vicuna 13B base from Hugging Face
    - ``MISTRAL_7B_HF``: LLaVA v1.6 with Mistral 7B base from Hugging Face

    Example:
        >>> from anomalib.models.image.vlm_ad.utils import ModelName
        >>> model_name = ModelName.LLAMA_OLLAMA  # doctest: +SKIP
        >>> model_name.value
        'llava'

    See Also:
        - :class:`VlmAd`: Main model class using these model options
        - :mod:`.backends`: Backend implementations for different models
    """

    LLAMA_OLLAMA = "llava"
    GPT_4O_MINI = "gpt-4o-mini"
    VICUNA_7B_HF = "llava-hf/llava-v1.6-vicuna-7b-hf"
    VICUNA_13B_HF = "llava-hf/llava-v1.6-vicuna-13b-hf"
    MISTRAL_7B_HF = "llava-hf/llava-v1.6-mistral-7b-hf"
