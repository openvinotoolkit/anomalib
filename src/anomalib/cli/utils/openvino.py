"""Utils for OpenVINO parser."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from jsonargparse import ArgumentParser

from anomalib.utils.exceptions import try_import

logger = logging.getLogger(__name__)


if try_import("openvino"):
    from openvino.tools.ovc.cli_parser import get_common_cli_parser
else:
    get_common_cli_parser = None


def add_openvino_export_arguments(parser: ArgumentParser) -> None:
    """Add OpenVINO arguments to parser under --mo key."""
    if get_common_cli_parser is not None:
        group = parser.add_argument_group("OpenVINO Model Optimizer arguments (optional)")
        mo_parser = get_common_cli_parser()
        # remove redundant keys from mo keys
        for arg in mo_parser._actions:  # noqa: SLF001
            if arg.dest in ("help", "input_model", "output_dir"):
                continue
            group.add_argument(f"--mo_args.{arg.dest}", type=arg.type, default=arg.default, help=arg.help)
    else:
        logger.info("OpenVINO is possibly not installed in the environment. Skipping adding it to parser.")
