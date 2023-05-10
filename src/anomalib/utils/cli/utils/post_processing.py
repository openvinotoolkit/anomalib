"""Post Processing utils"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from pytorch_lightning.cli import LightningArgumentParser

from anomalib.post_processing import NormalizationMethod, ThresholdMethod


def add_post_processing_arguments(parser: LightningArgumentParser):
    """Adds post processing arguments to the parser

    Args:
        parser (LightningArgumentParser): parser to add arguments to
    """
    post_processing_group = parser.add_argument_group(
        "Post Processing", description="Post Processing and Normalization Arguments"
    )
    post_processing_group.add_argument(
        "--post_processing.threshold_method",
        type=ThresholdMethod,
        default=ThresholdMethod.ADAPTIVE,
        help="Thresholding method",
    )
    post_processing_group.add_argument(
        "--post_processing.normalization_method",
        type=NormalizationMethod,
        default=NormalizationMethod.MIN_MAX,
        help="Normalization method",
    )
    post_processing_group.add_argument(
        "--post_processing.manual_image_threshold",
        type=Optional[float],
        required=False,
        help="Manual threshold for image",
    )
    post_processing_group.add_argument(
        "--post_processing.manual_pixel_threshold",
        type=Optional[float],
        required=False,
        help="Manual threshold for pixel",
    )
    parser.link_arguments("post_processing.manual_image_threshold", "trainer.manual_image_threshold")
    parser.link_arguments("post_processing.manual_pixel_threshold", "trainer.manual_pixel_threshold")
    parser.link_arguments("post_processing.threshold_method", "trainer.threshold_method")
    parser.link_arguments("post_processing.normalization_method", "trainer.normalization_method")
