"""Provides Troch inference."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import warnings
from pathlib import Path

import torch
from jsonargparse import ArgumentParser

from anomalib.data import TaskType
from anomalib.data.utils import generate_output_image_filename, get_image_filenames, read_image
from anomalib.deploy import TorchInferencer
from anomalib.post_processing import VisualizationMode, Visualizer


def get_torch_parser() -> ArgumentParser:
    """Get parser.

    Returns:
        ArgumentParser: The parser object.
    """
    parser = ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights")
    parser.add_argument("--input", type=Path, required=True, help="Path to an image to infer.")
    parser.add_argument("--output", type=Path, required=False, help="Path to save the output image.")
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="auto",
        help="Device to use for inference. Defaults to auto.",
        choices=["auto", "cpu", "gpu", "cuda"],  # cuda and gpu are the same but provided for convenience
    )
    parser.add_argument("--task", type=TaskType, required=False, help="Task type.", default=TaskType.CLASSIFICATION)
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


class TorchInference:
    """Torch inference.

    Args:
        weights (Path): Path to model weights.
        input (Path): Path to an image to infer.
        output (Path): Path to save the output image.
        device (str): Device to use for inference. Defaults to auto.
        task (TaskType): Task type.
        visualization_mode (VisualizationMode): Visualization mode.
        show (bool): Show the visualized predictions on the screen.
    """

    def __init__(
        self,
        weights: Path,
        input: Path,
        output: Path,
        device: str,
        task: TaskType,
        visualization_mode: VisualizationMode,
        show: bool,
    ):
        self.weights = weights
        self.input = input
        self.output = output
        self.device = device
        self.task = task
        self.visualization_mode = visualization_mode
        self.show = show

        torch.set_grad_enabled(False)
        self.inferencer = TorchInferencer(path=self.weights, device=self.device)
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
