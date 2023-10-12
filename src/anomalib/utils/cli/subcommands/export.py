"""Export utilities for Anomalib CLI"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from pathlib import Path

import torch
from jsonargparse import ActionConfigFile, Namespace
from jsonargparse._actions import _ActionSubCommands
from lightning.pytorch.cli import LightningArgumentParser
from openvino.tools.mo.utils.cli_parser import get_common_cli_parser

from anomalib.config.config import get_configurable_parameters
from anomalib.data.utils.transform import get_transforms
from anomalib.deploy import export_to_onnx, export_to_openvino, export_to_torch, get_metadata
from anomalib.models import get_model

logger = logging.getLogger(__name__)


def add_torch_export_arguments(subcommand: _ActionSubCommands):
    """Add torch parser to subcommand."""
    parser = _get_export_parser("torch")
    subcommand.add_subcommand("torch", parser)


def add_onnx_export_arguments(subcommand: _ActionSubCommands):
    """Add onnx parser to subcommand."""
    parser = _get_export_parser("ONNX")
    subcommand.add_subcommand("onnx", parser)


def add_openvino_export_arguments(subcommand: _ActionSubCommands):
    """Add OpenVINO parser to subcommand."""
    parser = _get_export_parser("OpenVINO")
    group = parser.add_argument_group("OpenVINO Model Optimizer arguments (optional)")
    mo_parser = get_common_cli_parser()
    # remove redundant keys from mo keys
    for arg in mo_parser._actions:  # noqa: SLF001
        if arg.dest in ("help", "input_model", "output_dir"):
            continue
        group.add_argument(f"--mo.{arg.dest}", type=arg.type, default=arg.default, help=arg.help)
    subcommand.add_subcommand("openvino", parser)


def _get_export_parser(subcommand: str):
    """Get the parser with common params for all the export subcommands."""
    parser = LightningArgumentParser(description=f"Export to {subcommand} format")
    parser.add_argument("--weights", type=Path, help="Path to the checkpoint file.", required=True)
    parser.add_argument("--model_config", type=Path, help="Path to the model config.", required=True)
    parser.add_argument("--export_path", type=Path, help="Path to save the exported model.")
    parser.add_argument(
        "-c",
        "--config",
        action=ActionConfigFile,
        help="Path to a configuration file in json or yaml format.",
    )
    return parser


def run_export(config: Namespace):
    """Run the export method.

    Args:
        config (Namespace): Parsed namespace
    """
    export_mode = config.export.export_mode
    config = config.export[export_mode]
    # Create export directory.
    model_config = get_configurable_parameters(config_path=config.model_config)
    export_path = config.export_path if config.export_path else model_config.trainer.default_root_dir
    export_path = Path(export_path) / "weights" / export_mode
    export_path.mkdir(parents=True, exist_ok=True)
    model = get_model(model_config)
    model.load_state_dict(torch.load(config.weights)["state_dict"])
    # get eval transform config if available otherwise get train config.
    # If both are not available then it is set to None
    transform_config = (
        model_config.data.init_args.transform_config_eval
        if model_config.data.init_args.transform_config_eval
        else model_config.data.init_args.transform_config_train
    )
    transforms = get_transforms(
        config=transform_config,
        image_size=model_config.data.init_args.get("image_size", None),
        center_crop=model_config.data.init_args.get("center_crop", None),
        normalization=model_config.data.init_args.normalization,
    )
    # Get metadata.
    metadata = get_metadata(model_config.task, transform=transforms.to_dict(), model=model)
    if export_mode == "torch":
        export_to_torch(model=model, metadata=metadata, export_path=export_path)
    elif export_mode in ("onnx", "openvino"):
        # Write metadata to json file. The file is written in the same directory as the target model.
        with (Path(export_path) / "metadata.json").open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, ensure_ascii=False, indent=4)
        onnx_path = export_to_onnx(
            model=model,
            input_size=model_config.model.init_args.input_size,
            export_path=export_path,
        )
        if export_mode == "openvino":
            export_to_openvino(
                export_path=export_path,
                input_model=onnx_path,
                metadata=metadata,
                input_size=model_config.model.init_args.input_size,
                **config.mo,
            )
    else:
        logger.exception(f"Unknown export mode {export_mode}")
        raise ValueError
