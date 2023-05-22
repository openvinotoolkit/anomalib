"""Provides Gradio inference."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module
from pathlib import Path

import gradio
import numpy as np
from jsonargparse import ArgumentParser

from anomalib.deploy import Inferencer


def get_gradio_parser() -> ArgumentParser:
    """Get command line arguments.

    Example:

        Example for Torch Inference.
        >>> python tools/inference/gradio_inference.py  \
        ...     --weights ./results/padim/mvtec/bottle/weights/torch/model.pt

    Returns:
        ArgumentParser: Argument parser for gradio inference.
    """
    parser = ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights")
    parser.add_argument("--metadata", type=Path, required=False, help="Path to a JSON file containing the metadata.")
    parser.add_argument("--share", type=bool, required=False, default=False, help="Share Gradio `share_url`")

    return parser


class GradioInference:
    """Gradio inference.

    Args:
        weights (Path): Path to model weights.
        metadata (Path): Path to a JSON file containing the metadata.
        share (bool, optional): Share Gradio `share_url`. Defaults to False.
    """

    def __init__(self, weights: Path, metadata: Path, share: bool = False):
        self.weights = weights
        self.metadata = metadata
        self.share = share

        self.inferencer = self.get_inferencer()

    def get_inferencer(self) -> Inferencer:
        """Parse args and open inferencer.

        Raises:
            ValueError: If unsupported model weight is passed.

        Returns:
            Inferencer: Torch or OpenVINO inferencer.
        """

        # Get the inferencer. We use .ckpt extension for Torch models and (onnx, bin)
        # for the openvino models.
        extension = self.weights.suffix
        inferencer: Inferencer
        module = import_module("anomalib.deploy")
        if extension in (".pt", ".pth", ".ckpt"):
            torch_inferencer = getattr(module, "TorchInferencer")
            inferencer = torch_inferencer(path=self.weights)

        elif extension in (".onnx", ".bin", ".xml"):
            if self.metadata is None:
                raise ValueError("When using OpenVINO Inferencer, the following arguments are required: --metadata")

            openvino_inferencer = getattr(module, "OpenVINOInferencer")
            inferencer = openvino_inferencer(path=self.weights, metadata_path=self.metadata)

        else:
            raise ValueError(
                f"Model extension is not supported. Torch Inferencer exptects a .ckpt file,"
                f"OpenVINO Inferencer expects either .onnx, .bin or .xml file. Got {extension}"
            )

        return inferencer

    def _infer(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Inference function, return anomaly map, score, heat map, prediction mask ans visualisation.

        Args:
            image (np.ndarray): image to compute

        Returns:
            tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
            heat_map, pred_mask, segmentation result.
        """
        # Perform inference for the given image.
        predictions = self.inferencer.predict(image=image)
        return (predictions.heat_map, predictions.pred_mask, predictions.segmentations)

    def run(self):
        """Run inference."""

        interface = gradio.Interface(
            self._infer,
            inputs=gradio.inputs.Image(
                shape=None, image_mode="RGB", source="upload", tool="editor", type="numpy", label="Image"
            ),
            outputs=[
                gradio.outputs.Image(type="numpy", label="Predicted Heat Map"),
                gradio.outputs.Image(type="numpy", label="Predicted Mask"),
                gradio.outputs.Image(type="numpy", label="Segmentation Result"),
            ],
            title="Anomalib",
            description="Anomalib Gradio",
        )

        interface.launch(share=self.share)
