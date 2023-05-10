"""Logging utils"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union

from pytorch_lightning.cli import LightningArgumentParser


def add_logging_arguments(parser: LightningArgumentParser):
    """Adds logging arguments to the parser

    Args:
        parser (LightningArgumentParser): parser to add arguments to
    """
    logging_group = parser.add_argument_group("logging", description="Logging Arguments")
    logging_group.add_argument("--logging.log_graph", type=bool, default=False, help="Log model graph to the logger(s)")
    logging_group.add_argument("--logging.loggers", type=Union[dict, List[dict]])
