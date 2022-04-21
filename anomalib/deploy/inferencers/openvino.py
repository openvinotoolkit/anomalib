"""This module contains inference-related abstract class and its Torch and OpenVINO implementations."""

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

from importlib.util import find_spec
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from omegaconf import DictConfig, ListConfig

from anomalib.pre_processing import PreProcessor

from .base import Inferencer

if find_spec("openvino") is not None:
    from openvino.inference_engine import (  # type: ignore  # pylint: disable=no-name-in-module
        IECore,
    )


class OpenVINOInferencer(Inferencer):
    """OpenVINO implementation for the inference.

    Args:
        config (DictConfig): Configurable parameters that are used
            during the training stage.
        path (Union[str, Path]): Path to the openvino onnx, xml or bin file.
        meta_data_path (Union[str, Path], optional): Path to metadata file. Defaults to None.
    """

    def __init__(
        self,
        config: Union[DictConfig, ListConfig],
        path: Union[str, Path, Tuple[bytes, bytes]],
        meta_data_path: Union[str, Path] = None,
    ):
        self.config = config
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
        config = self.config.transform if "transform" in self.config.keys() else None
        image_size = tuple(self.config.dataset.image_size)
        pre_processor = PreProcessor(config, image_size)
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
    ) -> Tuple[np.ndarray, float]:
        """Post process the output predictions.

        Args:
            predictions (np.ndarray): Raw output predicted by the model.
            meta_data (Dict, optional): Meta data. Post-processing step sometimes requires
                additional meta data such as image shape. This variable comprises such info.
                Defaults to None.

        Returns:
            np.ndarray: Post processed predictions that are ready to be visualized.
        """
        if meta_data is None:
            meta_data = self.meta_data

        predictions = predictions[self.output_blob]
        anomaly_map = predictions.squeeze()
        pred_score = anomaly_map.reshape(-1).max()

        anomaly_map, pred_score = self._normalize(anomaly_map, pred_score, meta_data)

        if "image_shape" in meta_data and anomaly_map.shape != meta_data["image_shape"]:
            anomaly_map = cv2.resize(anomaly_map, meta_data["image_shape"])

        return anomaly_map, float(pred_score)
