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

import os
from argparse import ArgumentParser, Namespace

import cv2

from anomalib.config import get_configurable_parameters
from anomalib.deploy.inference import Inferencer, OpenVINOInferencer, TorchInferencer


def get_args() -> Namespace:
    """Get command line arguments.

    Returns:
        Namespace: List of arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--model_config_path", type=str, required=True, help="Path to a model config file")
    parser.add_argument("--weight_path", type=str, required=True, help="Path to a model weights")
    parser.add_argument("--image_path", type=str, required=True, help="Path to an image to infer.")
    parser.add_argument("--save_path", type=str, required=False, help="Path to save the output image.")
    parser.add_argument("--meta_data", type=str, required=False, help="Path to JSON file containing the metadata.")

    return parser.parse_args()


def infer() -> None:
    """Perform inference on an input image."""

    # Get the command line arguments, and config from the config.yaml file.
    # This config file is also used for training and contains all the relevant
    # information regarding the data, model, train and inference details.
    args = get_args()
    config = get_configurable_parameters(model_config_path=args.model_config_path)

    # Get the inferencer. We use .ckpt extension for Torch models and (onnx, bin)
    # for the openvino models.
    extension = os.path.splitext(args.weight_path)[-1]
    inference: Inferencer
    if extension in (".ckpt"):
        inference = TorchInferencer(config=config, path=args.weight_path, meta_data_path=args.meta_data)

    elif extension in (".onnx", ".bin", ".xml"):
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

    # Show or save the output image, depending on what's provided as
    # the command line argument.
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    if args.save_path is None:
        cv2.imshow("Anomaly Map", output)
    else:
        cv2.imwrite(filename=args.save_path, img=output)


if __name__ == "__main__":
    infer()
