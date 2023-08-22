"""Anomalib OpenVINO Inferencer Script.

This script performs OpenVINO inference by reading a model from
file system, and show the visualization results.
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np

try:
    from openvino.model_api.models import AnomalyDetection, AnomalyResult
except ImportError as exception:
    raise ImportError("Ensure that OpenVINO model zoo is installed in your environment.") from exception


class Visualizer:
    @staticmethod
    def superimpose_pred_boxes(image: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
        """Superimpose predicted boxes on image.

        Args:
            image (np.ndarray): Input image.
            pred_boxes (np.ndarray): Predicted boxes.

        Returns:
            np.ndarray: Image with superimposed boxes.
        """
        for box in pred_boxes:
            if len(box) == 4:
                image = cv2.rectangle(
                    image,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    (0, 255, 0),
                    2,
                )
        return image

    @staticmethod
    def superimpose_anomaly_map(anomaly_map: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Superimpose anomaly map on the input image.

        Args:
            anomaly_map (np.ndarray): Anomaly map.
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Superimposed image.
        """
        # convert to color map
        if anomaly_map.max() <= 1.0:
            anomaly_map = (anomaly_map * 255).astype(np.uint8)
        anomaly_map = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)

        superimposed_map = cv2.addWeighted(image, 0.5, anomaly_map, 0.5, 0)
        return superimposed_map

    @staticmethod
    def superimpose_pred_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Superimpose predicted mask on image.

        Args:
            image (np.ndarray): Input image.
            mask (np.ndarray): Predicted mask.

        Returns:
            np.ndarray: Image with superimposed mask.
        """
        if mask.max() <= 1.0:
            mask *= 255
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            image = cv2.drawContours(image, contours, -1, (0, 0, 255), 5)
        return image

    def visualize(self, image: np.ndarray, outputs: AnomalyResult) -> np.ndarray:
        """Simple visualization of the inference results.

        Results are superimposed over each other.


        Args:
            image (np.ndarray): Input image.
            outputs (AnomalyResult): Inference results.

        Returns:
            np.ndarray: Visualized image.
        """
        if outputs.anomaly_map is not None:
            image = self.superimpose_anomaly_map(outputs.anomaly_map, image)
        if outputs.pred_mask is not None:
            self.superimpose_pred_mask(image, outputs.pred_mask)
        if outputs.pred_label is not None:
            pred_label = outputs.pred_label
            pred_label += f" ({outputs.pred_score:.2f})"
            image = cv2.putText(
                image,
                pred_label,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        if outputs.pred_boxes is not None:
            self.superimpose_pred_boxes(image, outputs.pred_boxes)
        return image


class ImageReader:
    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

    def __init__(self, image_path: str | Path):
        """Image reader.

        Args:
            image_path (str | Path): Path to the image or folder containing image.
        """
        self.image_path = Path(image_path)

    def __iter__(self) -> Iterator[Tuple[Path, np.ndarray]]:
        """Iterate over the image path.

        Yields:
            Path: Path to the image.
            np.ndarray: Image as numpy array.
        """
        if self.image_path.is_file():
            yield self.image_path, cv2.imread(str(self.image_path))

        if self.image_path.is_dir():
            for path in self.image_path.glob("**/*"):
                if path.suffix in self.IMG_EXTENSIONS:
                    yield path, cv2.imread(str(path))


def get_parser() -> ArgumentParser:
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
        help="Hardware device on which the model will be deployed",
        default="CPU",
        choices=["CPU", "GPU", "VPU"],
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
    weights_path = args.weights.with_suffix(".xml")
    model = AnomalyDetection.create_model(model=weights_path, device=args.device)
    visualizer = Visualizer()

    images = ImageReader(args.input)

    for image_path, image in images:
        prediction: AnomalyResult = model(image)
        output = visualizer.visualize(image, prediction)

        if args.output is None and args.show is False:
            warnings.warn(
                "Neither output path is provided nor show flag is set. Inferencer will run but return nothing."
            )

        if args.output:
            if args.output.is_dir():
                file_path = args.output / image_path.parent.name / image_path.name
            else:
                file_path = args.output
            cv2.imwrite(str(file_path), output)

        # Show the image in case the flag is set by the user.
        if args.show:
            cv2.imshow("Output Image", output)


if __name__ == "__main__":
    args = get_parser().parse_args()
    infer(args)
