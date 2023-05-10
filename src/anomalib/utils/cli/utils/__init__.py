"""CLI utils"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from .logging import add_logging_arguments
from .metrics import add_metrics_arguments
from .post_processing import add_post_processing_arguments
from .visualization import add_visualization_arguments

__all__ = [
    "add_logging_arguments",
    "add_metrics_arguments",
    "add_post_processing_arguments",
    "add_visualization_arguments",
]
