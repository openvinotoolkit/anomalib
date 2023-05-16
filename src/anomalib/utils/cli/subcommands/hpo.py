"""HPO subcommand."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from jsonargparse import ActionConfigFile, ArgumentParser, Namespace
from omegaconf import OmegaConf

from anomalib.utils.hpo import CometSweep, WandbSweep


def add_hpo_parser(parser: ArgumentParser):
    """Method that instantiates the argument parser."""
    sub_parser = ArgumentParser("HPO sweep")
    sub_parser.add_argument(
        "-c", "--config", action=ActionConfigFile, help="Path to a configuration file in json or yaml format."
    )
    parser._subcommands_action.add_subcommand("hpo", sub_parser, help="Perform HPO sweep using either Comet or Wandb")
    subparsers = sub_parser.add_subcommands(dest="backend", help="Sweep Backend")
    subparsers.add_subcommand("wandb", get_parser(), help="HPO sweep using Wandb")
    subparsers.add_subcommand("comet", get_parser(), help="HPO sweep using Comet")


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="HPO sweep using Wandb")
    parser.add_argument(
        "-c", "--config", action=ActionConfigFile, help="Path to a configuration file in json or yaml format."
    )
    parser.add_argument("--model_config", type=str, required=True, help="Path to a model config file")
    parser.add_argument("--sweep_config", type=str, required=True, help="Path to sweep configuration")
    parser.add_argument(
        "--entity",
        type=str,
        required=False,
        help="Username or workspace where you want to send your runs to. If not set, the default workspace is used.",
    )
    return parser


def run_hpo(config: Namespace):
    """Method that runs the hpo."""
    backend = config.backend
    model_config = OmegaConf.load(config[backend].model_config)
    hpo_config = OmegaConf.load(config[backend].sweep_config)
    entity = config[backend].get("entity", None)
    match config.backend:
        case "wandb":
            WandbSweep(model_config, hpo_config, entity=entity).run()
        case "comet":
            CometSweep(model_config, hpo_config, entity=entity).run()
        case _:
            raise ValueError(f"Unknown backend {config.backend}")
