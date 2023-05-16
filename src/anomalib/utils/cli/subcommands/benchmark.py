"""Benchmark subcommand."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from jsonargparse import ArgumentParser, Namespace
from omegaconf import OmegaConf

from anomalib.utils.benchmarking import distribute


def add_benchmarking_parser(parser: ArgumentParser):
    """Method that instantiates the argument parser."""
    sub_parser = ArgumentParser("Benchmark models")
    parser._subcommands_action.add_subcommand("benchmark", sub_parser, help="Perform Benchmarking")
    sub_parser.add_argument("--config", required=True, help="Path to a configuration file in yaml format.", type=str)


def run_benchmarking(config: Namespace):
    """Method that runs the benchmarking."""
    config = OmegaConf.load(config.config)
    distribute(config)
