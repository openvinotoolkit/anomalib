"""Visualization utils"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pytorch_lightning.cli import LightningArgumentParser

from anomalib.post_processing.visualizer import VisualizationMode
from anomalib.trainer.utils.visualization_manager import VisualizationStage


def add_visualization_arguments(parser: LightningArgumentParser):
    """Adds visualization arguments to the parser

    Args:
        parser (LightningArgumentParser): parser to add arguments to
    """
    visualization_group = parser.add_argument_group("Visualization", description="Visualization Arguments")
    visualization_group.add_argument(
        "--visualization.show_images", type=bool, default=False, help="Show images on the screen"
    )
    visualization_group.add_argument(
        "--visualization.log_images", type=bool, default=True, help="Log images to the logger(s)"
    )
    visualization_group.add_argument(
        "--visualization.visualization_mode",
        type=VisualizationMode,
        default=VisualizationMode.FULL,
        help="Visualization mode",
    )
    visualization_group.add_argument(
        "--visualization.visualization_stage",
        type=VisualizationStage,
        default=VisualizationStage.TEST,
        help="Stage at which to log images",
    )
    parser.link_arguments("visualization.show_images", "trainer.show_images")
    parser.link_arguments("visualization.log_images", "trainer.log_images")
    parser.link_arguments("visualization.visualization_mode", "trainer.visualization_mode")
    parser.link_arguments("visualization.visualization_stage", "trainer.visualization_stage")
