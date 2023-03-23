"""Anomalib Inferencer Script.

This script performs inference by reading a model config file from
command line, and show the visualization results.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path

from anomalib.data.utils import (
    generate_output_image_filename,
    get_image_filenames,
    read_image,
)
from anomalib.deploy import OpenVINOInferencer
from anomalib.post_processing import Visualizer


def get_args() -> Namespace:
    """Get command line arguments.

    Returns:
        Namespace: List of arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to a config file")
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

    args = parser.parse_args()

    return args


def infer() -> None:
    """Infer predictions.

    Show/save the output if path is to an image. If the path is a directory, go over each image in the directory.
    """
    # Get the command line arguments, and config from the config.yaml file.
    # This config file is also used for training and contains all the relevant
    # information regarding the data, model, train and inference details.
    args = get_args()

    # Get the inferencer.
    inferencer = OpenVINOInferencer(path=args.weights, metadata_path=args.metadata, device=args.device)
    visualizer = Visualizer(mode=args.visualization_mode, task=args.task)

    filenames = get_image_filenames(path=args.input)
    for filename in filenames:
        image = read_image(filename)
        predictions = inferencer.predict(image=image)
        output = visualizer.visualize_image(predictions)

        if args.output is None and args.show is False:
            warnings.warn(
                "Neither output path is provided nor show flag is set. Inferencer will run but return nothing."
            )

        if args.output:
            file_path = generate_output_image_filename(input_path=filename, output_path=args.output)
            visualizer.save(file_path=file_path, image=output)

        # Show the image in case the flag is set by the user.
        if args.show:
            visualizer.show(title="Output Image", image=output)


if __name__ == "__main__":
    infer()
