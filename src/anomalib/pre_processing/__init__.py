"""Pre-processing module for anomaly detection pipelines.

This module provides functionality for pre-processing data before model training
and inference through the :class:`PreProcessor` class.

The pre-processor handles:
    - Applying transforms to data during different pipeline stages
    - Managing stage-specific transforms (train/val/test)
    - Integrating with both PyTorch and Lightning workflows

Example:
    >>> from anomalib.pre_processing import PreProcessor
    >>> from torchvision.transforms.v2 import Resize
    >>> pre_processor = PreProcessor(transform=Resize(size=(256, 256)))
    >>> transformed_batch = pre_processor(batch)

The pre-processor is implemented as both a :class:`torch.nn.Module` and
:class:`lightning.pytorch.Callback` to support both inference and training
workflows.
"""

# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .pre_processor import PreProcessor

__all__ = ["PreProcessor"]
