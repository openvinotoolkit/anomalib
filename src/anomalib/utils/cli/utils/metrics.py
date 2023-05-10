"""Metrics configuration utils"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Union

from pytorch_lightning.cli import LightningArgumentParser
from torchmetrics import Metric


def add_metrics_arguments(parser: LightningArgumentParser):
    """Adds metrics arguments to the parser

    Args:
        parser (LightningArgumentParser): parser to add arguments to
    """
    metrics_group = parser.add_argument_group("Metrics", description="Metrics configuration arguments")
    metrics_group.add_argument("--metrics.image", type=Optional[Union[List[str], List[Metric]]], help="Image metrics")
    metrics_group.add_argument("--metrics.pixel", type=Optional[Union[List[str], List[Metric]]], help="Pixel metrics")
    parser.link_arguments("metrics.image", "trainer.image_metrics")
    parser.link_arguments("metrics.pixel", "trainer.pixel_metrics")
