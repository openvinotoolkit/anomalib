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
from omegaconf import DictConfig, ListConfig, OmegaConf
from openvino.inference_engine import IECore  # pylint: disable=no-name-in-module
from torch import Tensor, nn

from anomalib.core.model import AnomalyModule
from anomalib.data.transforms.pre_process import PreProcessor
from anomalib.data.utils import read_image
from anomalib.models import get_model
from anomalib.utils.normalization.cdf import normalize as normalize_cdf
from anomalib.utils.normalization.cdf import standardize
from anomalib.utils.normalization.min_max import normalize as normalize_min_max
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

    def predict(
        self, image: Union[str, np.ndarray], superimpose: bool = True, meta_data: Optional[dict] = None
    ) -> np.ndarray:
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
        if meta_data is None:
            if hasattr(self, "meta_data"):
                meta_data = getattr(self, "meta_data")
            else:
                meta_data = {}
        if isinstance(image, str):
            image = read_image(image)
        meta_data["image_shape"] = image.shape[:2]

        processed_image = self.pre_process(image)
        predictions = self.forward(processed_image)
        anomaly_map, _ = self.post_process(predictions, meta_data=meta_data)

        if superimpose is True:
            anomaly_map = superimpose_anomaly_map(anomaly_map, image)

        return anomaly_map

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Call predict on the Image.

        Args:
            image (np.ndarray): Input Image

        Returns:
            np.ndarray: Output predictions to be visualized
        """
        return self.predict(image)

    def _normalize(
        self,
        anomaly_maps: Union[Tensor, np.ndarray],
        pred_scores: Union[Tensor, np.float32],
        meta_data: Union[Dict, DictConfig],
    ) -> Tuple[Union[np.ndarray, Tensor], float]:
        """Applies normalization and resizes the image.

        Args:
            anomaly_maps (Union[Tensor, np.ndarray]): Predicted raw anomaly map.
            pred_scores (Union[Tensor, np.float32]): Predicted anomaly score
            meta_data (Dict): Meta data. Post-processing step sometimes requires
                additional meta data such as image shape. This variable comprises such info.
                Defaults to {}.

        Returns:
            Tuple[Union[np.ndarray, Tensor], float]: Post processed predictions that are ready to be visualized and
                predicted scores.


        """

        # min max normalization
        if "min" in meta_data and "max" in meta_data:
            anomaly_maps = normalize_min_max(
                anomaly_maps, meta_data["pixel_threshold"], meta_data["min"], meta_data["max"]
            )
            pred_scores = normalize_min_max(
                pred_scores, meta_data["image_threshold"], meta_data["min"], meta_data["max"]
            )

        # standardize pixel scores
        if "pixel_mean" in meta_data.keys() and "pixel_std" in meta_data.keys():
            anomaly_maps = standardize(
                anomaly_maps, meta_data["pixel_mean"], meta_data["pixel_std"], center_at=meta_data["image_mean"]
            )
            anomaly_maps = normalize_cdf(anomaly_maps, meta_data["pixel_threshold"])

        # standardize image scores
        if "image_mean" in meta_data.keys() and "image_std" in meta_data.keys():
            pred_scores = standardize(pred_scores, meta_data["image_mean"], meta_data["image_std"])
            pred_scores = normalize_cdf(pred_scores, meta_data["image_threshold"])

        if isinstance(anomaly_maps, Tensor):
            anomaly_maps = anomaly_maps.numpy()

        if "image_shape" in meta_data and anomaly_maps.shape != meta_data["image_shape"]:
            anomaly_maps = cv2.resize(anomaly_maps, meta_data["image_shape"])

        return anomaly_maps, float(pred_scores)

    def _load_meta_data(self, path: Optional[Union[str, Path]] = None) -> Union[DictConfig, Dict]:
        meta_data = {}
        if path is not None:
            meta_data = OmegaConf.load(path)
        return meta_data


class TorchInferencer(Inferencer):
    """PyTorch implementation for the inference.

    Args:
        config (DictConfig): Configurable parameters that are used
            during the training stage.
        path (Union[str, Path]): Path to the model ckpt file.
        meta_data_path (Union[str, Path], optional): Path to metadata file. If none, it tries to load the params
                from the model state_dict. Defaults to None.
    """

    def __init__(
        self,
        config: Union[DictConfig, ListConfig],
        path: Union[str, Path, AnomalyModule],
        meta_data_path: Union[str, Path] = None,
    ):
        self.config = config
        if isinstance(path, AnomalyModule):
            self.model = path
        else:
            self.model = self.load_model(path)

        self.meta_data = self._load_meta_data(meta_data_path)

    def _load_meta_data(self, path: Optional[Union[str, Path]] = None) -> Union[Dict, DictConfig]:
        """Load metadata from file or from model state dict.

        Args:
            path (Optional[Union[str, Path]], optional): Path to metadata file. If none, it tries to load the params
                from the model state_dict. Defaults to None.

        Returns:
            Dict: Dictionary containing the meta_data.
        """
        meta_data = {}
        model_params = self.model.state_dict()
        meta_data_params = {
            "image_threshold.value": "image_threshold",
            "pixel_threshold.value": "pixel_threshold",
            "training_distribution.pixel_mean": "pixel_mean",
            "training_distribution.image_mean": "image_mean",
            "training_distribution.pixel_std": "pixel_std",
            "training_distribution.image_std": "image_std",
            "min_max.min": "min",
            "min_max.max": "max",
        }
        if path is None:
            for param, key in meta_data_params.items():
                if param in model_params.keys():
                    val = model_params[param].to(self.model.device)
                    # Skip adding the value to metadata if value if undefined.
                    if not np.isinf(val.numpy()).all():
                        meta_data[key] = val
        else:
            meta_data = super()._load_meta_data(path)
        return meta_data

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

    def post_process(
        self, predictions: Tensor, meta_data: Optional[Union[Dict, DictConfig]] = None
    ) -> Tuple[np.ndarray, float]:
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
            meta_data = self.meta_data

        if isinstance(predictions, Tensor):
            anomaly_map = predictions
            pred_score = anomaly_map.reshape(-1).max()
        else:
            anomaly_map, pred_score = predictions
            pred_score = pred_score.detach().cpu().numpy()

        anomaly_map = anomaly_map.squeeze()

        anomaly_map, pred_score = self._normalize(anomaly_map, pred_score, meta_data)

        return anomaly_map, float(pred_score)


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
                Defaults to {}.

        Returns:
            np.ndarray: Post processed predictions that are ready to be visualized.
        """
        if meta_data is None:
            meta_data = self.meta_data

        predictions = predictions[self.output_blob]
        anomaly_map = predictions.squeeze()
        pred_score = anomaly_map.reshape(-1).max()

        anomaly_map, pred_score = self._normalize(anomaly_map, pred_score, meta_data)

        return anomaly_map, pred_score
