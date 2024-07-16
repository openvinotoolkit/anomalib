"""Subcommand for pipelines."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from jsonargparse import Namespace

from anomalib.cli.utils.help_formatter import get_short_docstring
from anomalib.utils.exceptions import try_import

logger = logging.getLogger(__name__)

if try_import("anomalib.pipelines"):
    from anomalib.pipelines import Benchmark
    from anomalib.pipelines.components.base import Pipeline

    PIPELINE_REGISTRY: dict[str, type[Pipeline]] | None = {"benchmark": Benchmark}
else:
    PIPELINE_REGISTRY = None


def pipeline_subcommands() -> dict[str, dict[str, str]]:
    """Return subcommands for pipelines."""
    if PIPELINE_REGISTRY is not None:
        return {name: {"description": get_short_docstring(pipeline)} for name, pipeline in PIPELINE_REGISTRY.items()}
    return {}


def run_pipeline(args: Namespace) -> None:
    """Run pipeline."""
    logger.warning("This feature is experimental. It may change or be removed in the future.")
    if PIPELINE_REGISTRY is not None:
        subcommand = args.subcommand
        config = args[subcommand]
        PIPELINE_REGISTRY[subcommand]().run(config)
    else:
        msg = "Pipeline is not available"
        raise ValueError(msg)
