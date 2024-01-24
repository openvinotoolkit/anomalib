"""WinCLIP Model."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .lightning_model import WinClip
from .torch_model import WinClipModel

__all__ = ["WinClip", "WinClipModel"]
