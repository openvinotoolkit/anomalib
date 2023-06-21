"""This module contains inference-related abstract class and its Torch and OpenVINO implementations."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from anomalib.data.utils import read_image  # TODO remove this to avoid anomalib dependency
from anomalib.post_processing import ImageResult  # TODO remove this to avoid anomalib dependency

if find_spec("openvino") is not None:
    from openvino.runtime import Core
else:
    raise ImportError("OpenVINO is not installed. Please install OpenVINO to use OpenVINOInferencer.")

LABEL_MAPPING = {0: "normal", 1: "anomaly"}


class OpenVINOInferencer:
    """OpenVINO implementation for the inference.

    Args:
        path (str | Path): Path to the openvino onnx, xml or bin file.
        device (str | None, optional): Device to run the inference on. Defaults to "CPU".
        task (tar): Task type. Defaults to classification.
        config (dict | None, optional): Config for OpenVINO's compile_model. Defaults to None.
    """

    def __init__(
        self,
        path: str | Path | tuple[bytes, bytes],
        device: str | None = "CPU",
        task: str = "classification",
        config: dict | None = None,
    ) -> None:
        self.device = device

        self.config = config
        self.input_blob, self.output_blobs, self.model = self.load_model(path)
        self.task = task

    def load_model(self, path: str | Path | tuple[bytes, bytes]):
        """Load the OpenVINO model.

        Args:
            path (str | Path | tuple[bytes, bytes]): Path to the onnx or xml and bin files
                                                        or tuple of .xml and .bin data as bytes.

        Returns:
            [tuple[str, str, ExecutableNetwork]]: Input and Output blob names
                together with the Executable network.
        """
        ie_core = Core()
        # If tuple of bytes is passed

        if isinstance(path, tuple):
            model = ie_core.read_model(model=path[0], weights=path[1], init_from_buffer=True)
        else:
            path = path if isinstance(path, Path) else Path(path)
            if path.suffix in (".bin", ".xml"):
                if path.suffix == ".bin":
                    bin_path, xml_path = path, path.with_suffix(".xml")
                elif path.suffix == ".xml":
                    xml_path, bin_path = path, path.with_suffix(".bin")
                model = ie_core.read_model(xml_path, bin_path)
            elif path.suffix == ".onnx":
                model = ie_core.read_model(path)
            else:
                raise ValueError(f"Path must be .onnx, .bin or .xml file. Got {path.suffix}")
        # Create cache folder
        cache_folder = Path("cache")
        cache_folder.mkdir(exist_ok=True)
        ie_core.set_property({"CACHE_DIR": cache_folder})

        compile_model = ie_core.compile_model(model=model, device_name=self.device, config=self.config)

        input_blob = compile_model.input(0)
        output_blobs = compile_model.outputs

        return input_blob, output_blobs, compile_model

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        """Pre process the input image by applying transformations.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: pre-processed image.
        """

        if len(image.shape) == 3:
            if list(self.input_blob.partial_shape)[2:] != list(image.shape)[:-1]:
                image = cv2.resize(
                    image,
                    dsize=(
                        int(self.input_blob.partial_shape[3].to_string()),
                        int(self.input_blob.partial_shape[2].to_string()),
                    ),
                    interpolation=cv2.INTER_LINEAR,
                )
            image = np.expand_dims(np.transpose(image, (2, 0, 1)), axis=0)  # 1 x 3 x H x W
        return image.astype(np.int64)

    def forward(self, image: np.ndarray) -> np.ndarray:
        """Forward-Pass input tensor to the model.

        Args:
            image (np.ndarray): Input tensor.

        Returns:
            np.ndarray: Output predictions.
        """
        return self.model(image)

    def post_process(self, predictions: np.ndarray, image_shape: tuple[int, int]) -> dict[str, Any]:
        """Post process the output predictions.

        Args:
            predictions (np.ndarray): Raw output predicted by the model.
            image_shape (tuple[int, int]): Original image shape.

        Returns:
            dict[str, Any]: Post processed prediction results.
        """
        output = {}
        for output_blob in self.output_blobs:
            output[output_blob.any_name] = predictions[output_blob]
        return output

    @staticmethod
    def _get_boxes(mask: np.ndarray) -> np.ndarray:
        """Get bounding boxes from masks.

        Args:
            masks (np.ndarray): Input mask of shape (H, W)

        Returns:
            np.ndarray: array of shape (N, 4) containing the bounding box coordinates of the objects in the masks
            in xyxy format.
        """
        _, comps = cv2.connectedComponents(mask)

        labels = np.unique(comps)
        boxes = []
        for label in labels[labels != 0]:
            y_loc, x_loc = np.where(comps == label)
            boxes.append([np.min(x_loc), np.min(y_loc), np.max(x_loc), np.max(y_loc)])
        boxes = np.stack(boxes) if boxes else np.empty((0, 4))
        return boxes

    def predict(
        self,
        image: str | Path | np.ndarray,
    ) -> ImageResult:
        """Perform a prediction for a given input image.

        The main workflow is (i) pre-processing, (ii) forward-pass, (iii) post-process.

        Args:
            image (Union[str, np.ndarray]): Input image whose output is to be predicted.
                It could be either a path to image or numpy array itself.

            metadata: Metadata information such as shape, threshold.

        Returns:
            ImageResult: Prediction results to be visualized.
        """
        if isinstance(image, (str, Path)):
            image_arr: np.ndarray = read_image(image)
        else:  # image is already a numpy array. Kept for mypy compatibility.
            image_arr = image
        image_shape = image_arr.shape[:2]

        processed_image = self.pre_process(image_arr)
        predictions = self.forward(processed_image)
        output = self.post_process(predictions, image_shape=image_shape)

        image_arr = cv2.resize(image_arr, dsize=(output["anomaly_map"].shape[1], output["anomaly_map"].shape[0]))

        return ImageResult(
            image=image_arr,
            pred_score=output.get("pred_score", None),
            pred_label=output.get("pred_label", None),
            anomaly_map=output.get("anomaly_map", None),
            pred_mask=output.get("pred_mask", None),
            pred_boxes=output.get("pred_boxes", None),
            box_labels=output.get("box_labels", None),
        )

    def __call__(self, image: np.ndarray) -> ImageResult:
        """Call predict on the Image.

        Args:
            image (np.ndarray): Input Image

        Returns:
            ImageResult: Prediction results to be visualized.
        """
        return self.predict(image)
