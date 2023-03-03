"""These hooks are attached at different stages of the training/validation/prediction loop."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .postprocessing import PostProcessingHooks

__all__ = ["PostProcessingHooks"]
