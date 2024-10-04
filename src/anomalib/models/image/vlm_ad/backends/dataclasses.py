"""Dataclasses."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass
class Prompt:
    """Ollama prompt."""

    few_shot: str
    predict: str
