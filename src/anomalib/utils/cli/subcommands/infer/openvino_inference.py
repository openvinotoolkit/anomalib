"""Provides OpenVINO inference."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import warnings
from pathlib import Path

from jsonargparse import ArgumentParser

from anomalib.data import TaskType
from anomalib.data.utils import generate_output_image_filename, get_image_filenames, read_image
from anomalib.deploy import OpenVINOInferencer
from anomalib.post_processing import VisualizationMode, Visualizer


def get_openvino_parser():
    """Get OpenVINO parser.

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
        type=TaskType,
        required=False,
        help="Task type.",
        default=TaskType.CLASSIFICATION,
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
        type=VisualizationMode,
        required=False,
        default=VisualizationMode.SIMPLE,
        help="Visualization mode.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        required=False,
        help="Show the visualized predictions on the screen.",
    )

    return parser


class OpenVINOInference:
    """OpenVINO inference.

    Args:
        weights (Path): Path to model weights.
        metadata (Path): Path to a JSON file containing the metadata.
        input (Path): Path to an image to infer.
        output (Path): Path to save the output image.
        task (TaskType): Task type.
        device (str): Hardware device on which the model will be deployed.
        visualization_mode (VisualizationMode): Visualization mode.
        show (bool): Show the visualized predictions on the screen.
    """

    def __init__(
        self,
        weights: Path,
        metadata: Path,
        input: Path,
        output: Path,
        task: TaskType,
        device: str,
        visualization_mode: VisualizationMode,
        show: bool,
    ):
        self.weights = weights
        self.metadata = metadata
        self.input = input
        self.output = output
        self.task = task
        self.device = device
        self.visualization_mode = visualization_mode
        self.show = show

        self.inferencer = OpenVINOInferencer(path=self.weights, metadata_path=self.metadata, device=self.device)
        self.visualizer = Visualizer(mode=self.visualization_mode, task=self.task)

    def run(self):
        """Run inference."""
        filenames = get_image_filenames(path=self.input)
        for filename in filenames:
            image = read_image(filename)
            predictions = self.inferencer.predict(image=image)
            output = self.visualizer.visualize_image(predictions)

            if self.output is None and self.show is False:
                warnings.warn(
                    "Neither output path is provided nor show flag is set. Inferencer will run but return nothing."
                )

            if self.output:
                file_path = generate_output_image_filename(input_path=filename, output_path=self.output)
                self.visualizer.save(file_path=file_path, image=output)

            # Show the image in case the flag is set by the user.
            if self.show:
                self.visualizer.show(title="Output Image", image=output)
