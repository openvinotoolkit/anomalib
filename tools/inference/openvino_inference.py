"""Anomalib OpenVINO Inferencer Script.

This script performs OpenVINO inference by reading a model from
file system, and show the visualization results.
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from anomalib.data.utils import generate_output_image_filename, get_image_filenames, read_image
from anomalib.data.utils.image import save_image, show_image
from anomalib.deploy import OpenVINOInferencer
from anomalib.utils.visualization import ImageVisualizer

logger = logging.getLogger(__name__)


def get_parser() -> ArgumentParser:
    """Get parser.

    Returns:
        ArgumentParser: The parser object.
    """
    parser = ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to a JSON file containing the metadata.")
    parser.add_argument("--input", type=Path, required=True, help="Path to an image to infer.")
    parser.add_argument("--output", type=Path, required=False, help="Path to save the output image.")
    parser.add_argument(
        "--task",
        type=str,
        required=False,
        help="Task type.",
        default="classification",
        choices=["classification", "detection", "segmentation"],
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        help="Hardware device on which the model will be deployed",
        default="CPU",
        choices=["CPU", "GPU", "VPU"],
    )
    parser.add_argument(
        "--visualization_mode",
        type=str,
        required=False,
        default="simple",
        help="Visualization mode.",
        choices=["full", "simple"],
    )
    parser.add_argument(
        "--show",
        action="store_true",
        required=False,
        help="Show the visualized predictions on the screen.",
    )

    return parser


def infer(args: Namespace) -> None:
    """Infer predictions.

    Show/save the output if path is to an image. If the path is a directory, go over each image in the directory.

    Args:
        args (Namespace): The arguments from the command line.
    """
    # Get the inferencer.
    inferencer = OpenVINOInferencer(path=args.weights, metadata=args.metadata, device=args.device)
    visualizer = ImageVisualizer(mode=args.visualization_mode, task=args.task)

    filenames = get_image_filenames(path=args.input)
    for filename in filenames:
        image = read_image(filename)
        predictions = inferencer.predict(image=image)
        output = visualizer.visualize_image(predictions)

        if args.output is None and args.show is False:
            msg = "Neither output path is provided nor show flag is set. Inferencer will run but return nothing."
            logger.warning(msg)

        if args.output:
            file_path = generate_output_image_filename(input_path=filename, output_path=args.output)
            save_image(filename=file_path, image=output)

        # Show the image in case the flag is set by the user.
        if args.show:
            show_image(title="Output Image", image=output)


if __name__ == "__main__":
    args = get_parser().parse_args()
    infer(args)
