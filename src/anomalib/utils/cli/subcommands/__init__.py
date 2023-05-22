"""Subcommands for Anomalib CLI."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .benchmark import add_benchmarking_parser, run_benchmarking
from .export import add_export_parser, run_export
from .hpo import add_hpo_parser, run_hpo
from .infer import add_inference_parser, run_inference

__all__ = [
    "add_benchmarking_parser",
    "add_export_parser",
    "add_hpo_parser",
    "add_inference_parser",
    "run_benchmarking",
    "run_export",
    "run_hpo",
    "run_inference",
]
