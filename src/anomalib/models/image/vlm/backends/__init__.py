"""VLM backends."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .base import Backend
from .ollama import Ollama

__all__ = ["Backend", "Ollama"]
