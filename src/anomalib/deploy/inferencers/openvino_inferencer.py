"""This module contains OpenVINO inference.

This is meant to be standalone and not depend on any other anomalib modules.
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import openvino.runtime as ov


class OpenVINOInferencer:
    """OpenVINO inferencer.

    Args:
        weights (Path): Path to the onnx or xml and bin files
        device (str): Device to use for inference. See OpenVINO's documentation for available devices.
    """

    def __init__(
        self,
        weights: Path,
        device: str = "CPU",
    ) -> None:
        self.device = device
        self.model = self.load_model(weights)
        self.input_blob = self.model.input(0)
        self.image_size = self.get_input_shape()
        self.output_blobs = self.model.outputs

    def load_model(self, path: Path) -> ov.CompiledModel:
        """Load model from file.

        Args:
            path (Path): Path to the onnx or xml and bin files

        Raises:
            ValueError: Unsupported model format

        Returns:
            ov.CompiledModel: OpenVINO compiled model
        """
        core = ov.Core()
        if path.suffix == ".xml":
            model = core.read_model(str(path))
        elif path.suffix == ".bin":
            model = core.read_model(str(path.with_suffix(".xml")))
        elif path.suffix == ".onnx":
            model = core.read_model(str(path))
        else:
            raise ValueError(f"Unsupported model format: {path.suffix}")
        compiled_model = core.compile_model(model, self.device)
        return compiled_model

    def get_input_shape(self) -> tuple[int, int]:
        """Gets the input shape of the model."""
        # assumes model input to have shape [B, C, H, W]
        height = int(self.input_blob.partial_shape[2].to_string())
        width = int(self.input_blob.partial_shape[3].to_string())
        return (height, width)

    def predict(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Perform inference on image

        Args:
            image (np.ndarray): Input image
        """
        outputs = self.model(image)
        # Convert OV predictions to dict of names and values
        outputs = {output_blob.any_name: outputs[output_blob] for output_blob in self.output_blobs}
        return outputs
