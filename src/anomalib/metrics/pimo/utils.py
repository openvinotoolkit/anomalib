"""Torch-oriented interfaces for `utils.py`."""

# Original Code
# https://github.com/jpcbertoldo/aupimo
#
# Modified
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import torch

logger = logging.getLogger(__name__)


def images_classes_from_masks(masks: torch.Tensor) -> torch.Tensor:
    """Deduce the image classes from the masks."""
    return (masks == 1).any(axis=(1, 2)).to(torch.int32)
