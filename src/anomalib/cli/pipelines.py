"""Subcommand for pipelines."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from jsonargparse import ArgumentParser, Namespace

from anomalib.utils.exceptions import try_import

if try_import("anomalib.pipelines.pipeline"):
    from anomalib.pipelines import Pipeline, PoolExecutor
    from anomalib.pipelines.jobs import BenchmarkJob

    PIPELINE_REGISTRY: dict[str, Pipeline] | None = {
        "benchmark": Pipeline(PoolExecutor(BenchmarkJob())),
    }
else:
    PIPELINE_REGISTRY = None


def add_pipeline_subparsers(parser: ArgumentParser) -> None:
    """Add subparsers for pipelines."""
    if PIPELINE_REGISTRY is not None:
        subcommands = parser.add_subcommands(dest="subcommand", help="Run Pipelines", required=True)
        for name, pipeline in PIPELINE_REGISTRY.items():
            subcommands.add_subcommand(name, pipeline.get_parser(), help=f"Run {name} pipeline")


def run_pipeline(args: Namespace) -> None:
    """Run pipeline."""
    if PIPELINE_REGISTRY is not None:
        config = args.pipeline[args.pipeline.subcommand]
        PIPELINE_REGISTRY[args.pipeline.subcommand].run(config)
    else:
        msg = "Pipeline is not available"
        raise ValueError(msg)
