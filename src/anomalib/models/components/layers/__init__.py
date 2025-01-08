"""Neural network layers used in anomaly detection models.

This module provides custom neural network layer implementations that can be used
as building blocks in anomaly detection models.

Classes:
    SSPCAB: Spatial-Spectral Pixel-Channel Attention Block layer that combines
        spatial and channel attention mechanisms.

Example:
    >>> import torch
    >>> from anomalib.models.components.layers import SSPCAB
    >>> # Create attention layer
    >>> attention = SSPCAB(in_channels=64)
    >>> # Apply attention to input tensor
    >>> input_tensor = torch.randn(1, 64, 32, 32)
    >>> output = attention(input_tensor)
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .sspcab import SSPCAB

__all__ = ["SSPCAB"]
