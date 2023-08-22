"""Anomalib Gradio Script.

This script provide a gradio web interface
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from argparse import ArgumentParser
from importlib import import_module
from pathlib import Path

import cv2
import gradio as gr
import gradio.inputs
import gradio.outputs
import numpy as np
from openvino.model_api.models import AnomalyDetection, AnomalyResult
from skimage.segmentation import mark_boundaries

from anomalib.deploy import TorchInferencer


def get_parser() -> ArgumentParser:
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
    parser.add_argument("--share", type=bool, required=False, default=False, help="Share Gradio `share_url`")

    return parser


def get_inferencer(weight_path: Path) -> AnomalyDetection | TorchInferencer:
    """Parse args and open inferencer.

    Args:
        weight_path (Path): Path to model weights.

    Raises:
        ValueError: If unsupported model weight is passed.

    Returns:
        Inferencer: Torch or OpenVINO inferencer.
    """

    # Get the inferencer. We use .ckpt extension for Torch models and (onnx, bin)
    # for the openvino models.
    extension = weight_path.suffix
    inferencer: TorchInferencer | AnomalyDetection
    module = import_module("anomalib.deploy")
    if extension in (".pt", ".pth", ".ckpt"):
        torch_inferencer = getattr(module, "TorchInferencer")
        inferencer = torch_inferencer(path=weight_path)

    elif extension in (".bin", ".xml"):
        inferencer = AnomalyDetection.create_model(weight_path.with_suffix(".xml"))
    else:
        raise ValueError(
            f"Model extension is not supported. Torch Inferencer exptects a .ckpt file,"
            f"OpenVINO Inferencer expects either .bin or .xml file. Got {extension}"
        )

    return inferencer


def infer(
    image: np.ndarray, inferencer: TorchInferencer | AnomalyDetection
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inference function, return anomaly map, score, heat map, prediction mask ans visualisation.

    Args:
        image (np.ndarray): image to compute
        inferencer (Inferencer): model inferencer

    Returns:
        tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        heat_map, pred_mask, segmentation result.
    """
    # Perform inference for the given image.
    predictions: tuple[np.ndarray, np.ndarray, np.ndarray]
    if isinstance(inferencer, TorchInferencer):
        result = inferencer.predict(image=image)
        predictions = (result.heat_map, result.pred_mask, result.segmentations)
    else:
        # openvino requires bgr image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result = inferencer(image)
        predictions = _convert_from_modelapi(image, result)
    return predictions


def _convert_from_modelapi(image: np.ndarray, result: AnomalyResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert the result from model api to the format that gradio can handle."""

    segmentations: np.ndarray = mark_boundaries(image, result.pred_mask, color=(0, 0, 1), mode="thick")
    if segmentations.max() <= 1.0:
        segmentations = (segmentations * 255).astype(np.uint8)
        segmentations = cv2.cvtColor(segmentations, cv2.COLOR_RGB2BGR)

    anomaly_map = result.anomaly_map if result.anomaly_map is not None else np.zeros_like(image)
    if anomaly_map.max() <= 1.0:
        anomaly_map = (anomaly_map * 255).astype(np.uint8)
    anomaly_map = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(image, 0.5, anomaly_map, 0.5, 0)
    superimposed = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

    pred_mask = result.pred_mask if result.pred_mask is not None else np.zeros_like(image)
    if pred_mask.max() <= 1.0:
        pred_mask = (pred_mask * 255).astype(np.uint8)

    return superimposed, pred_mask, segmentations


if __name__ == "__main__":
    args = get_parser().parse_args()
    gradio_inferencer = get_inferencer(args.weights)

    interface = gr.Interface(
        fn=lambda image: infer(image, gradio_inferencer),
        inputs=[
            gradio.inputs.Image(
                shape=None, image_mode="RGB", source="upload", tool="editor", type="numpy", label="Image"
            ),
        ],
        outputs=[
            gradio.outputs.Image(type="numpy", label="Predicted Heat Map"),
            gradio.outputs.Image(type="numpy", label="Predicted Mask"),
            gradio.outputs.Image(type="numpy", label="Segmentation Result"),
        ],
        title="Anomalib",
        description="Anomalib Gradio",
    )

    interface.launch(share=args.share)
