"""
This module contains inference-related abstract class
and its Torch and OpenVINO implementations.
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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


from anomalib.core.model import AnomalyModule
from anomalib.data.transforms.pre_process import PreProcessor
from anomalib.data.utils import read_image
from anomalib.models import get_model
from anomalib.utils.post_process import superimpose_anomaly_map

from anomaly_classification.configs.configuration import BaseAnomalyClassificationConfig

import cv2

import numpy as np

from omegaconf import DictConfig, ListConfig

from openvino.model_zoo.model_api.adapters import OpenvinoAdapter, create_core
from openvino.model_zoo.model_api.models import Model

from ote_sdk.entities.annotation import AnnotationSceneEntity
from ote_sdk.entities.label import LabelEntity
from ote_sdk.usecases.exportable_code.prediction_to_annotation_converter import (
    AnomalyClassificationToAnnotationConverter,
    IPredictionToAnnotationConverter
)

import torch
from torch import Tensor, nn


class Inferencer(ABC):
    """
    Abstract class for the inference.
    This is used by both Torch and OpenVINO inference.
    """

    @property
    @abstractmethod
    def converter(self):
        pass

    @abstractmethod
    def pre_process(self, image: np.ndarray) -> Tuple[Any, Any]:
        """
        This method should pre-process the input image, and return the processed output
        and if required a Tuple with metadata that is required for post_process to work.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, image: Any) -> Any:
        """
        NOTE: The input is typed as Any at the moment, mainly because it could be numpy
            array,torch Tensor or tf Tensor. In the future, it could be an idea to be
            more specific.

        This method should perform the prediction by forward-passing the input image
        to the model, and return the predictions in a dictionary format.

        For instance, for a segmentation task, the predictions could be {"mask": mask}.

        """
        raise NotImplementedError

    @abstractmethod
    def post_process(self, prediction: Any, metadata: Any) -> np.ndarray:
        """
        This method should include the post-processing methods that are applied to the
        raw predictions from the self.forward() stage.
        """
        raise NotImplementedError

    def predict(self, image: Union[str, np.ndarray], superimpose: bool = True) -> AnnotationSceneEntity:
        """
        Perform a prediction for a given input image. The main workflow is
        (i) pre-processing, (ii) forward-pass, (iii) post-process.

        Args:
            image (Union[str, np.ndarray]): Input image whose output is to be predicted.
                It could be either a path to image or numpy array itself.

            superimpose (bool): If this is set to True, output predictions
                will be superimposed onto the original image. If false, `predict`
                method will return the raw heatmap.

        Returns:
            AnnotationSceneEntity: Output annotation scene to be visualized.
        """

        if isinstance(image, str):
            image = read_image(image)

        processed_image, metadata = self.pre_process(image)
        predictions = self.forward(processed_image)
        output = self.post_process(predictions, metadata)

        if superimpose is True:
            output = superimpose_anomaly_map(output, image)

        return self.converter.convert_to_annotation(output, metadata)

    def __call__(self, image: np.ndarray) -> AnnotationSceneEntity:
        """
        Call predict on the Image.
        Args:
            image (np.ndarray): Input Image
        Returns:
            AnnotationSceneEntity: Output annotation scene to be visualized.
        """
        return self.predict(image)


class TorchInferencer(Inferencer):
    """
    PyTorch implementation for the inference.

    Args:
        config (DictConfig): Configurable parameters that are used
            during the training stage.
        labels: List of labels that was used during model training.
        path (Union[str, Path]): Path to the model ckpt file.
    """

    def __init__(
        self,
        config: Union[DictConfig, ListConfig],
        labels: List[LabelEntity],
        path: Union[str, Path, AnomalyModule]
    ):
        self.config = config
        if isinstance(path, AnomalyModule):
            self.model = path
        else:
            self.model = self.load_model(path)
        self._converter = AnomalyClassificationToAnnotationConverter(labels)

    @property
    def converter(self) -> IPredictionToAnnotationConverter:
        return self._converter

    def load_model(self, path: Union[str, Path]) -> nn.Module:
        """
        Load the PyTorch model.

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
        """
        Pre process the input image by applying transformations.

        Args:
            image (np.ndarray): Input image

        Returns:
            Tensor: pre-processed image.
        """
        pre_processor = PreProcessor(config=self.config.transform, to_tensor=True)
        processed_image = pre_processor(image=image)["image"]

        if len(processed_image) == 3:
            processed_image = processed_image.unsqueeze(0)

        return processed_image

    def forward(self, image: Tensor) -> Tensor:
        """
        Forward-Pass input tensor to the model.

        Args:
            image (Tensor): Input tensor.

        Returns:
            Tensor: Output predictions.
        """
        return self.model(image)

    def post_process(self, predictions: Tensor, metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Post process the output predictions.

        Args:
            predictions (Tensor): Raw output predicted by the model.
            meta_data (Dict, optional): Meta data. Post-processing step sometimes requires
                additional meta data such as image shape. This variable comprises such info.
                Defaults to {}.

        Returns:
            np.ndarray: Post processed predictions that are ready to be visualized.
        """

        if metadata is None:
            metadata = {}

        predictions = predictions.squeeze().detach().numpy()

        if "image_shape" in metadata and predictions.shape != metadata["image_shape"]:
            predictions = cv2.resize(predictions, metadata["image_shape"])

        return predictions


class OpenVINOInferencer(Inferencer):
    """
    OpenVINO implementation for the inference.

    Args:
        config (DictConfig): Configurable parameters that are used
            during the training stage.
        hparams: Hyper parameters that the model should use.
        threshold
        labels: List of labels that was used during model training.
        path (Union[str, Path]): Path to the openvino onnx, xml or bin file
    """

    def __init__(
        self,
        config: Union[DictConfig, ListConfig],
        hparams: BaseAnomalyClassificationConfig,
        threshold: float,
        labels: List[LabelEntity],
        path: Union[str, Path, Tuple[bytes, bytes]]
    ):
        self.config = config
        self.labels = labels
        try:
            model_file = path[0] if isinstance(path, tuple) else path
            weight_file = path[1] if isinstance(path, tuple) else None
            model_adapter = OpenvinoAdapter(create_core(), model_path=model_file, weights_path=weight_file)
            label_names = [label.name for label in self.labels]

            self.configuration = {'mean_values': list(np.array(self.config.transform.normalize.mean) * 255),
                                  'scale_values': list(np.array(self.config.transform.normalize.std) * 255),
                                  'threshold': threshold,
                                  'labels': label_names}
            self.model = Model.create_model(hparams.inference_parameters.class_name.value,
                                            model_adapter, self.configuration)
            self.model.load()
        except ValueError as e:
            print(e)
        self._converter = AnomalyClassificationToAnnotationConverter(self.labels)

    @property
    def converter(self) -> IPredictionToAnnotationConverter:
        return self._converter

    def pre_process(self, image: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        return self.model.preprocess(image)

    def post_process(self, prediction: Dict[str, np.ndarray], metadata: Dict[str, Any]) -> np.ndarray:
        return self.model.postprocess(prediction, metadata)

    def forward(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self.model.infer_sync(inputs)
