"""Flow components used in anomaly detection models.

This module provides flow-based components that can be used in anomaly detection
models. These components help model complex data distributions and transformations.

Classes:
    AllInOneBlock: A block that combines multiple flow operations into a single
        transformation.

Example:
    >>> import torch
    >>> from anomalib.models.components.flow import AllInOneBlock
    >>> # Create flow block
    >>> flow = AllInOneBlock(channels=64)
    >>> # Apply flow transformation
    >>> x = torch.randn(1, 64, 32, 32)
    >>> y, logdet = flow(x)
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .all_in_one_block import AllInOneBlock

__all__ = ["AllInOneBlock"]
