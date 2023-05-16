"""Export subcommand."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module
from pathlib import Path

import torch
from jsonargparse import ActionConfigFile, ArgumentParser, Namespace
from openvino.tools.mo.utils.cli_parser import get_common_cli_parser

from anomalib.deploy import ExportMode
from anomalib.deploy.export import export


def add_export_parser(parser: ArgumentParser):
    """Method that instantiates the argument parser."""
    sub_parser = ArgumentParser()
    sub_parser.add_argument(
        "-c", "--config", action=ActionConfigFile, help="Path to a configuration file in json or yaml format."
    )
    parser._subcommands_action.add_subcommand(
        "export", sub_parser, help=f"Export model to one of {[mode.value for mode in set(ExportMode)]}"
    )
    subparsers = sub_parser.add_subcommands(dest="format", help="Export type")
    subparsers.add_subcommand("openvino", get_openvino_subparser(), help="Export to OpenVINO")
    subparsers.add_subcommand("onnx", get_subparser("ONNX"), help="Export to ONNX")
    subparsers.add_subcommand("torch", get_subparser("Torch"), help="Export to Torch")


def get_openvino_subparser() -> ArgumentParser:
    parser = get_subparser("OpenVINO")
    mo_parser = get_common_cli_parser()

    # Add exposed parameters to a separate group
    group = parser.add_argument_group("OpenVINO model optimizer arguments. These are optional")
    for arg in mo_parser._actions:
        if arg.dest not in ("help", "input_model", "output_dir", "extensions"):
            group.add_argument(f"--mo.{arg.dest}", type=arg.type, default=arg.default, help=arg.help, required=False)
    return parser


def get_subparser(format: str) -> ArgumentParser:
    """Creates a subparser for the given format.

    Args:
        format (str): Format to create a subparser for.
    """
    parser = ArgumentParser(description=f"Export model to {format}")
    parser.add_argument(
        "-c", "--config", action=ActionConfigFile, help="Path to a configuration file in json or yaml format."
    )
    parser.add_argument("--weights", type=Path, required=True, help="Path to model checkpoint weights")
    parser.add_argument(
        "--export_path", type=Path, required=False, help="Path to export model. Defaults to weights path"
    )
    return parser


def run_export(config: Namespace):
    """Run export subcommand."""
    format = config.format
    config = config[format]
    checkpoint = torch.load(config.weights)
    model_class = import_module(".".join(checkpoint["model_name"].split(".")[:-1]))
    model = getattr(model_class, checkpoint["model_name"].split(".")[-1])(**checkpoint["hyper_parameters"])
    model.load_state_dict(checkpoint["state_dict"])

    export_path = export(
        trainer=checkpoint,
        model=model,
        export_mode=ExportMode(format),
        input_size=checkpoint["hyper_parameters"]["input_size"],
        export_root=Path(config.export_path),
    )
    print(f"Model exported to {export_path}")
