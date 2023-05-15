"""Benchmark subcommand."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from jsonargparse import ArgumentParser

from anomalib.utils.benchmarking import distribute


def add_benchmarking_parser(parser: ArgumentParser):
    """Method that instantiates the argument parser."""
    sub_parser = ArgumentParser("Benchmark models")
    parser._subcommands_action.add_subcommand("benchmark", sub_parser, help="Perform Benchmarking")
    sub_parser.add_function_arguments(distribute, as_group=False)
