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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from openvino.inference_engine import IECore  # pylint: disable=no-name-in-module
from torch import Tensor, nn

from anomalib.core.model import AnomalyModule
from anomalib.data.transforms.pre_process import PreProcessor
from anomalib.data.utils import read_image
from anomalib.models import get_model
from anomalib.utils.post_process import superimpose_anomaly_map


class Inferencer(ABC):
    """Abstract class for the inference.

    This is used by both Torch and OpenVINO inference.
    """

    @abstractmethod
    def load_model(self, path: Union[str, Path]):
        """Load Model."""
        raise NotImplementedError

    @abstractmethod
    def pre_process(self, image: np.ndarray) -> Union[np.ndarray, Tensor]:
        """Pre-process."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, image: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
        """Forward-Pass input to model."""
        raise NotImplementedError

    @abstractmethod
    def post_process(self, predictions: Union[np.ndarray, Tensor], meta_data: Optional[Dict]) -> np.ndarray:
        """Post-Process."""
        raise NotImplementedError

    def predict(self, image: Union[str, np.ndarray], superimpose: bool = True) -> np.ndarray:
        """Perform a prediction for a given input image.

        The main workflow is (i) pre-processing, (ii) forward-pass, (iii) post-process.

        Args:
            image (Union[str, np.ndarray]): Input image whose output is to be predicted.
                It could be either a path to image or numpy array itself.

            superimpose (bool): If this is set to True, output predictions
                will be superimposed onto the original image. If false, `predict`
                method will return the raw heatmap.

        Returns:
            np.ndarray: Output predictions to be visualized.
        """
        if isinstance(image, str):
            image = read_image(image)

        processed_image = self.pre_process(image)
        predictions = self.forward(processed_image)
        output = self.post_process(predictions, meta_data={"image_shape": image.shape[:2]})

        if superimpose is True:
            output = superimpose_anomaly_map(output, image)

        return output

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Call predict on the Image.

        Args:
            image (np.ndarray): Input Image

        Returns:
            np.ndarray: Output predictions to be visualized
        """
        return self.predict(image)


class TorchInferencer(Inferencer):
    """PyTorch implementation for the inference.

    Args:
        config (DictConfig): Configurable parameters that are used
            during the training stage.
        path (Union[str, Path]): Path to the model ckpt file.
    """

    def __init__(self, config: Union[DictConfig, ListConfig], path: Union[str, Path, AnomalyModule]):
        self.config = config
        if isinstance(path, AnomalyModule):
            self.model = path
        else:
            self.model = self.load_model(path)

    def load_model(self, path: Union[str, Path]) -> nn.Module:
        """Load the PyTorch model.

        Args:
            path (Union[str, Path]): Path to model ckpt file.

        Returns:
            (nn.Module): PyTorch Lightning model.
        """
        model = get_model(self.config)
        model.load_state_dict(torch.load(path)["state_dict"])
        model.eval()
        return model

    def pre_process(self, image: np.ndarray) -> Tensor:
        """Pre process the input image by applying transformations.

        Args:
            image (np.ndarray): Input image

        Returns:
            Tensor: pre-processed image.
        """
        config = self.config.transform if "transform" in self.config.keys() else None
        image_size = tuple(self.config.dataset.image_size)
        pre_processor = PreProcessor(config, image_size)
        processed_image = pre_processor(image=image)["image"]

        if len(processed_image) == 3:
            processed_image = processed_image.unsqueeze(0)

        return processed_image

    def forward(self, image: Tensor) -> Tensor:
        """Forward-Pass input tensor to the model.

        Args:
            image (Tensor): Input tensor.

        Returns:
            Tensor: Output predictions.
        """
        return self.model(image)

    def post_process(self, predictions: Tensor, meta_data: Optional[Dict] = None) -> np.ndarray:
        """Post process the output predictions.

        Args:
            predictions (Tensor): Raw output predicted by the model.
            meta_data (Dict, optional): Meta data. Post-processing step sometimes requires
                additional meta data such as image shape. This variable comprises such info.
                Defaults to {}.

        Returns:
            np.ndarray: Post processed predictions that are ready to be visualized.
        """
        if meta_data is None:
            meta_data = {}

        predictions = predictions.squeeze().detach().numpy()

        if "image_shape" in meta_data and predictions.shape != meta_data["image_shape"]:
            predictions = cv2.resize(predictions, meta_data["image_shape"])

        return predictions


class OpenVINOInferencer(Inferencer):
    """OpenVINO implementation for the inference.

    Args:
        config (DictConfig): Configurable parameters that are used
            during the training stage.
        path (Union[str, Path]): Path to the openvino onnx, xml or bin file
    """

    def __init__(self, config: Union[DictConfig, ListConfig], path: Union[str, Path, Tuple[bytes, bytes]]):
        self.config = config
        self.input_blob, self.output_blob, self.network = self.load_model(path)

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

    def post_process(self, predictions: np.ndarray, meta_data: Optional[Dict] = None) -> np.ndarray:
        """Post process the output predictions.

        Args:
            predictions (np.ndarray): Raw output predicted by the model.
            meta_data (Dict, optional): Meta data. Post-processing step sometimes requires
                additional meta data such as image shape. This variable comprises such info.
                Defaults to {}.

        Returns:
            np.ndarray: Post processed predictions that are ready to be visualized.
        """
        if meta_data is None:
            meta_data = {}

        predictions = predictions[self.output_blob]
        predictions = predictions.squeeze()

        if "image_shape" in meta_data and predictions.shape != meta_data["image_shape"]:
            predictions = cv2.resize(predictions, meta_data["image_shape"])

        return predictions
