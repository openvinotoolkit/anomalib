"""Export utilities for Anomalib CLI."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from jsonargparse import ActionConfigFile
from jsonargparse._actions import _ActionSubCommands
from lightning.pytorch.cli import LightningArgumentParser

from anomalib.deploy import export_to_onnx, export_to_openvino, export_to_torch
from anomalib.utils.exceptions import try_import

logger = logging.getLogger(__name__)


if try_import("openvino"):
    from openvino.tools.mo.utils.cli_parser import get_common_cli_parser
else:
    get_common_cli_parser = None


def add_torch_export_arguments(subcommand: _ActionSubCommands) -> None:
    """Add torch parser to subcommand."""
    parser = _get_export_parser("torch")
    parser.add_function_arguments(export_to_torch)
    subcommand.add_subcommand("torch", parser, help="Export to torch format")


def add_onnx_export_arguments(subcommand: _ActionSubCommands) -> None:
    """Add onnx parser to subcommand."""
    parser = _get_export_parser("ONNX")
    parser.add_function_arguments(export_to_onnx)
    subcommand.add_subcommand("onnx", parser, help="Export to ONNX format")


def add_openvino_export_arguments(subcommand: _ActionSubCommands) -> None:
    """Add OpenVINO parser to subcommand."""
    if get_common_cli_parser is not None:
        parser = _get_export_parser("OpenVINO")
        parser.add_function_arguments(export_to_openvino, skip={"mo_args"})
        group = parser.add_argument_group("OpenVINO Model Optimizer arguments (optional)")
        mo_parser = get_common_cli_parser()
        # remove redundant keys from mo keys
        for arg in mo_parser._actions:  # noqa: SLF001
            if arg.dest in ("help", "input_model", "output_dir"):
                continue
            group.add_argument(f"--mo_args.{arg.dest}", type=arg.type, default=arg.default, help=arg.help)
        subcommand.add_subcommand("openvino", parser, help="Export to OpenVINO format")
    else:
        logger.info("OpenVINO is possibly not installed in the environment. Skipping adding it to parser.")


def _get_export_parser(subcommand: str) -> LightningArgumentParser:
    """Get the parser with common params for all the export subcommands."""
    parser = LightningArgumentParser(description=f"Export to {subcommand} format")
    parser.add_argument(
        "-c",
        "--config",
        action=ActionConfigFile,
        help="Path to a configuration file in json or yaml format.",
    )
    return parser
