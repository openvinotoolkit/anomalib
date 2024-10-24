"""Dataclasses."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum


@dataclass
class Prompt:
    """Prompt."""

    few_shot: str
    predict: str


class ModelName(Enum):
    """List of supported models."""

    LLAMA_OLLAMA = "llava"
    GPT_4O_MINI = "gpt-4o-mini"
    VICUNA_7B_HF = "llava-hf/llava-v1.6-vicuna-7b-hf"
    VICUNA_13B_HF = "llava-hf/llava-v1.6-vicuna-13b-hf"
    MISTRAL_7B_HF = "llava-hf/llava-v1.6-mistral-7b-hf"
