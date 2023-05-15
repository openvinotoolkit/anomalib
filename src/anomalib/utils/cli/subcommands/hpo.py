"""HPO subcommand."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from jsonargparse import ActionConfigFile, ArgumentParser

from anomalib.utils.hpo import CometSweep, WandbSweep


def add_hpo_parser(parser: ArgumentParser):
    """Method that instantiates the argument parser."""
    sub_parser = ArgumentParser()
    sub_parser.add_argument(
        "-c", "--config", action=ActionConfigFile, help="Path to a configuration file in json or yaml format."
    )
    parser._subcommands_action.add_subcommand("hpo", sub_parser, help="Perform HPO sweep using either Comet or Wandb")
    subparsers = sub_parser.add_subcommands(dest="hpo", help="Sweep Backend")
    subparsers.add_subcommand("wandb", get_wandb_parser(), help="HPO sweep using Wandb")
    subparsers.add_subcommand("comet", get_comet_parser(), help="HPO sweep using Comet")


def get_wandb_parser() -> ArgumentParser:
    parser = ArgumentParser(description="HPO sweep using Wandb")
    parser.add_argument(
        "-c", "--config", action=ActionConfigFile, help="Path to a configuration file in json or yaml format."
    )
    parser.add_class_arguments(WandbSweep, as_group=False)
    return parser


def get_comet_parser() -> ArgumentParser:
    parser = ArgumentParser(description="HPO sweep using Comet")
    parser.add_argument(
        "-c", "--config", action=ActionConfigFile, help="Path to a configuration file in json or yaml format."
    )
    parser.add_class_arguments(CometSweep, as_group=False)
    return parser
