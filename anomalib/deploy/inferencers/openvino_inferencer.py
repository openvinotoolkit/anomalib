"""This module contains inference-related abstract class and its Torch and OpenVINO implementations."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
from omegaconf import DictConfig, ListConfig

from anomalib.config import get_configurable_parameters
from anomalib.pre_processing import PreProcessor

from .base_inferencer import Inferencer

if find_spec("openvino") is not None:
    from openvino.inference_engine import (  # type: ignore  # pylint: disable=no-name-in-module
        IECore,
    )


class OpenVINOInferencer(Inferencer):
    """OpenVINO implementation for the inference.

    Args:
        config (Union[str, Path, DictConfig, ListConfig]): Configurable parameters that are used
            during the training stage.
        path (Union[str, Path]): Path to the openvino onnx, xml or bin file.
        meta_data_path (Union[str, Path], optional): Path to metadata file. Defaults to None.
    """

    def __init__(
        self,
        config: Union[str, Path, DictConfig, ListConfig],
        path: Union[str, Path, Tuple[bytes, bytes]],
        meta_data_path: Union[str, Path] = None,
    ):
        # Check and load the configuration
        if isinstance(config, (str, Path)):
            self.config = get_configurable_parameters(config_path=config)
        elif isinstance(config, (DictConfig, ListConfig)):
            self.config = config
        else:
            raise ValueError(f"Unknown config type {type(config)}")

        self.input_blob, self.output_blob, self.network = self.load_model(path)
        self.meta_data = super()._load_meta_data(meta_data_path)

    def load_model(self, path: Union[str, Path, Tuple[bytes, bytes]]):
        """Load the OpenVINO model.

        Args:
            path (Union[str, Path, Tuple[bytes, bytes]]): Path to the onnx or xml and bin files
                                                        or tuple of .xml and .bin data as bytes.

        Returns:
            [Tuple[str, str, ExecutableNetwork]]: Input and Output blob names
                together with the Executable network.
        """
        ie_core = IECore()
        # If tuple of bytes is passed

        if isinstance(path, tuple):
            network = ie_core.read_network(model=path[0], weights=path[1], init_from_buffer=True)
        else:
            path = path if isinstance(path, Path) else Path(path)
            if path.suffix in (".bin", ".xml"):
                if path.suffix == ".bin":
                    bin_path, xml_path = path, path.with_suffix(".xml")
                elif path.suffix == ".xml":
                    xml_path, bin_path = path, path.with_suffix(".bin")
                network = ie_core.read_network(xml_path, bin_path)
            elif path.suffix == ".onnx":
                network = ie_core.read_network(path)
            else:
                raise ValueError(f"Path must be .onnx, .bin or .xml file. Got {path.suffix}")

        input_blob = next(iter(network.input_info))
        output_blob = next(iter(network.outputs))
        executable_network = ie_core.load_network(network=network, device_name="CPU")

        return input_blob, output_blob, executable_network

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        """Pre process the input image by applying transformations.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: pre-processed image.
        """
        transform_config = (
            self.config.dataset.transform_config.val if "transform_config" in self.config.dataset.keys() else None
        )
        image_size = tuple(self.config.dataset.image_size)
        pre_processor = PreProcessor(transform_config, image_size)
        processed_image = pre_processor(image=image)["image"]

        if len(processed_image.shape) == 3:
            processed_image = np.expand_dims(processed_image, axis=0)

        if processed_image.shape[-1] == 3:
            processed_image = processed_image.transpose(0, 3, 1, 2)

        return processed_image

    def forward(self, image: np.ndarray) -> np.ndarray:
        """Forward-Pass input tensor to the model.

        Args:
            image (np.ndarray): Input tensor.

        Returns:
            np.ndarray: Output predictions.
        """
        return self.network.infer(inputs={self.input_blob: image})

    def post_process(
        self, predictions: np.ndarray, meta_data: Optional[Union[Dict, DictConfig]] = None
    ) -> Dict[str, Any]:
        """Post process the output predictions.

        Args:
            predictions (np.ndarray): Raw output predicted by the model.
            meta_data (Dict, optional): Meta data. Post-processing step sometimes requires
                additional meta data such as image shape. This variable comprises such info.
                Defaults to None.

        Returns:
            Dict[str, Any]: Post processed prediction results.
        """
        if meta_data is None:
            meta_data = self.meta_data

        predictions = predictions[self.output_blob]

        # Initialize the result variables.
        anomaly_map: Optional[np.ndarray] = None
        pred_label: Optional[float] = None
        pred_mask: Optional[float] = None

        # If predictions returns a single value, this means that the task is
        # classification, and the value is the classification prediction score.
        if len(predictions.shape) == 1:
            task = "classification"
            pred_score = predictions.item()
        else:
            task = "segmentation"
            anomaly_map = predictions.squeeze()
            pred_score = anomaly_map.reshape(-1).max()

        # Common practice in anomaly detection is to assign anomalous
        # label to the prediction if the prediction score is greater
        # than the image threshold.
        if "image_threshold" in meta_data:
            pred_label = pred_score >= meta_data["image_threshold"]

        if task == "segmentation":
            if "pixel_threshold" in meta_data:
                pred_mask = (anomaly_map >= meta_data["pixel_threshold"]).astype(np.uint8)

            anomaly_map, pred_score = self._normalize(anomaly_map, pred_score, meta_data)

            if "image_shape" in meta_data and anomaly_map.shape != meta_data["image_shape"]:
                image_height = meta_data["image_shape"][0]
                image_width = meta_data["image_shape"][1]
                anomaly_map = cv2.resize(anomaly_map, (image_width, image_height))

                if pred_mask is not None:
                    pred_mask = cv2.resize(pred_mask, (image_width, image_height))

        return {
            "anomaly_map": anomaly_map,
            "pred_label": pred_label,
            "pred_score": pred_score,
            "pred_mask": pred_mask,
        }
