"""Export subcommand."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from jsonargparse import ActionConfigFile, ArgumentParser
from openvino.tools.mo.utils.cli_parser import get_common_cli_parser

from anomalib.deploy import ExportMode
from anomalib.deploy.export import export_to_onnx, export_to_openvino, export_to_torch


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
    subparsers.add_subcommand("onnx", get_onnx_subparser(), help="Export to ONNX")
    subparsers.add_subcommand("torch", get_torch_subparser(), help="Export to Torch")


def get_openvino_subparser() -> ArgumentParser:
    parser = ArgumentParser(description="Export model to OpenVINO")
    parser.add_argument(
        "-c", "--config", action=ActionConfigFile, help="Path to a configuration file in json or yaml format."
    )
    parser.add_function_arguments(export_to_openvino, as_group=False)
    mo_parser = get_common_cli_parser()

    # Add exposed parameters to a separate group
    group = parser.add_argument_group("OpenVINO model optimizer arguments. These are optional")
    for arg in mo_parser._actions:
        if arg.dest not in ("help", "input_model", "output_dir"):
            group.add_argument(f"--mo.{arg.dest}", type=arg.type, default=arg.default, help=arg.help, required=False)
    return parser


def get_onnx_subparser() -> ArgumentParser:
    parser = ArgumentParser(description="Export model to ONNX")
    parser.add_argument(
        "-c", "--config", action=ActionConfigFile, help="Path to a configuration file in json or yaml format."
    )
    parser.add_function_arguments(export_to_onnx, as_group=False)
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    return parser


def get_torch_subparser() -> ArgumentParser:
    parser = ArgumentParser(description="Export model to Torch")
    parser.add_argument(
        "-c", "--config", action=ActionConfigFile, help="Path to a configuration file in json or yaml format."
    )
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_function_arguments(export_to_torch, as_group=False)
    return parser
