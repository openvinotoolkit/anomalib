"""Anomalib Inferencer Script.

This script performs inference by reading a model config file from
command line, and show the visualization results.
"""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from argparse import ArgumentParser, Namespace
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np

from anomalib.config import get_configurable_parameters
from anomalib.deploy.inferencers.base import Inferencer


def get_args() -> Namespace:
    """Get command line arguments.

    Returns:
        Namespace: List of arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--model_config_path", type=Path, required=True, help="Path to a model config file")
    parser.add_argument("--weight_path", type=Path, required=True, help="Path to a model weights")
    parser.add_argument("--image_path", type=Path, required=True, help="Path to an image to infer.")
    parser.add_argument("--save_path", type=Path, required=False, help="Path to save the output image.")
    parser.add_argument("--meta_data", type=Path, required=False, help="Path to JSON file containing the metadata.")

    return parser.parse_args()


def add_label(prediction: np.ndarray, scores: float, font: int = cv2.FONT_HERSHEY_PLAIN) -> np.ndarray:
    """If the model outputs score, it adds the score to the output image.

    Args:
        prediction (np.ndarray): Resized anomaly map.
        scores (float): Confidence score.

    Returns:
        np.ndarray: Image with score text.
    """
    text = f"Confidence Score {scores:.0%}"
    font_size = prediction.shape[1] // 1024 + 1  # Text scale is calculated based on the reference size of 1024
    (width, height), baseline = cv2.getTextSize(text, font, font_size, thickness=font_size // 2)
    label_patch = np.zeros((height + baseline, width + baseline, 3), dtype=np.uint8)
    label_patch[:, :] = (225, 252, 134)
    cv2.putText(label_patch, text, (0, baseline // 2 + height), font, font_size, 0)
    prediction[: baseline + height, : baseline + width] = label_patch
    return prediction


def infer() -> None:
    """Perform inference on an input image."""

    # Get the command line arguments, and config from the config.yaml file.
    # This config file is also used for training and contains all the relevant
    # information regarding the data, model, train and inference details.
    args = get_args()
    config = get_configurable_parameters(model_config_path=args.model_config_path)

    # Get the inferencer. We use .ckpt extension for Torch models and (onnx, bin)
    # for the openvino models.
    extension = args.weight_path.suffix
    inference: Inferencer
    if extension in (".ckpt"):
        module = import_module("anomalib.deploy.inferencers.torch")
        TorchInferencer = getattr(module, "TorchInferencer")  # pylint: disable=invalid-name
        inference = TorchInferencer(config=config, model_source=args.weight_path, meta_data_path=args.meta_data)

    elif extension in (".onnx", ".bin", ".xml"):
        module = import_module("anomalib.deploy.inferencers.openvino")
        OpenVINOInferencer = getattr(module, "OpenVINOInferencer")  # pylint: disable=invalid-name
        inference = OpenVINOInferencer(config=config, path=args.weight_path, meta_data_path=args.meta_data)

    else:
        raise ValueError(
            f"Model extension is not supported. Torch Inferencer exptects a .ckpt file,"
            f"OpenVINO Inferencer expects either .onnx, .bin or .xml file. Got {extension}"
        )

    # Perform inference for the given image or image path. if image
    # path is provided, `predict` method will read the image from
    # file for convenience. We set the superimpose flag to True
    # to overlay the predicted anomaly map on top of the input image.
    output = inference.predict(image=args.image_path, superimpose=True)

    # Incase both anomaly map and scores are returned add scores to the image.
    if isinstance(output, tuple):
        anomaly_map, score = output
        output = add_label(anomaly_map, score)

    # Show or save the output image, depending on what's provided as
    # the command line argument.
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    if args.save_path is None:
        cv2.imshow("Anomaly Map", output)
    else:
        cv2.imwrite(filename=str(args.save_path), img=output)


if __name__ == "__main__":
    infer()
