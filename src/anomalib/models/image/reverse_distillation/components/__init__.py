"""PyTorch modules for Reverse Distillation."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .bottleneck import get_bottleneck_layer
from .de_resnet import get_decoder

__all__ = ["get_bottleneck_layer", "get_decoder"]
