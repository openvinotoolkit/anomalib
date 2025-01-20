"""Anomalib pipeline subcommands.

This module provides functionality for managing and running Anomalib pipelines through
the CLI. It includes support for benchmarking and other pipeline operations.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from jsonargparse import Namespace
from lightning_utilities.core.imports import module_available

from anomalib.cli.utils.help_formatter import get_short_docstring

logger = logging.getLogger(__name__)

if module_available("anomalib.pipelines"):
    from anomalib.pipelines import Benchmark
    from anomalib.pipelines.components.base import Pipeline

    PIPELINE_REGISTRY: dict[str, type[Pipeline]] | None = {"benchmark": Benchmark}
else:
    PIPELINE_REGISTRY = None


def pipeline_subcommands() -> dict[str, dict[str, str]]:
    """Get available pipeline subcommands.

    Returns:
        dict[str, dict[str, str]]: Dictionary mapping subcommand names to their descriptions.

    Example:
        Pipeline subcommands are available only if the pipelines are installed::

        >>> pipeline_subcommands()
        {
            'benchmark': {
                'description': 'Run benchmarking pipeline for model evaluation'
            }
        }
    """
    if PIPELINE_REGISTRY is not None:
        return {name: {"description": get_short_docstring(pipeline)} for name, pipeline in PIPELINE_REGISTRY.items()}
    return {}


def run_pipeline(args: Namespace) -> None:
    """Run a pipeline with the provided arguments.

    Args:
        args (Namespace): Arguments for the pipeline, including the subcommand
            and configuration.

    Raises:
        ValueError: If pipelines are not available in the current installation.

    Note:
        This feature is experimental and may change or be removed in future versions.
    """
    logger.warning("This feature is experimental. It may change or be removed in the future.")
    if PIPELINE_REGISTRY is not None:
        subcommand = args.subcommand
        config = args[subcommand]
        PIPELINE_REGISTRY[subcommand]().run(config)
    else:
        msg = "Pipeline is not available"
        raise ValueError(msg)
