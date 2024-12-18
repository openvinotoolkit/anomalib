"""OpenVINO CLI utilities.

This module provides utilities for adding OpenVINO-specific arguments to the Anomalib CLI.
It handles the integration of OpenVINO Model Optimizer parameters into the command line interface.
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from jsonargparse import ArgumentParser
from lightning_utilities.core.imports import module_available

logger = logging.getLogger(__name__)


if module_available("openvino"):
    from openvino.tools.ovc.cli_parser import get_common_cli_parser
else:
    get_common_cli_parser = None


def add_openvino_export_arguments(parser: ArgumentParser) -> None:
    """Add OpenVINO Model Optimizer arguments to the parser.

    This function adds OpenVINO-specific export arguments to the parser under the `ov_args` prefix.
    If OpenVINO is not installed, it logs an informational message and skips adding the arguments.

    The function adds Model Optimizer arguments like data_type, mean_values, etc. as optional
    parameters that can be used during model export to OpenVINO format.

    Args:
        parser (ArgumentParser): The argument parser to add OpenVINO arguments to.
            This should be an instance of jsonargparse.ArgumentParser.

    Examples:
        Add OpenVINO arguments to a parser:

        >>> from jsonargparse import ArgumentParser
        >>> parser = ArgumentParser()
        >>> add_openvino_export_arguments(parser)

        The parser will now accept OpenVINO arguments like:

        >>> # parser.parse_args(['--ov_args.data_type', 'FP16'])
        >>> # parser.parse_args(['--ov_args.mean_values', '[123.675,116.28,103.53]'])

    Notes:
        - Requires OpenVINO to be installed to add the arguments
        - Automatically skips redundant arguments that are handled elsewhere:
            - help
            - input_model
            - output_dir
        - Arguments are added under the 'ov_args' prefix for namespacing
        - All OpenVINO arguments are made optional

    See Also:
        - OpenVINO Model Optimizer docs: https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html
        - OpenVINO Python API: https://docs.openvino.ai/latest/api/python_api.html
    """
    if get_common_cli_parser is not None:
        group = parser.add_argument_group("OpenVINO Model Optimizer arguments (optional)")
        ov_parser = get_common_cli_parser()
        # remove redundant keys from mo keys
        for arg in ov_parser._actions:  # noqa: SLF001
            if arg.dest in {"help", "input_model", "output_dir"}:
                continue
            group.add_argument(f"--ov_args.{arg.dest}", type=arg.type, default=arg.default, help=arg.help)
    else:
        logger.info("OpenVINO is possibly not installed in the environment. Skipping adding it to parser.")
