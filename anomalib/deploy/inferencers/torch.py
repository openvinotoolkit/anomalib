"""This module contains Torch inference implementations."""

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

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch import Tensor

from anomalib.core.model import AnomalyModule
from anomalib.data.transforms.pre_process import PreProcessor
from anomalib.deploy.optimize import get_model_metadata
from anomalib.models import get_model

from .base import Inferencer


class TorchInferencer(Inferencer):
    """PyTorch implementation for the inference.

    Args:
        config (DictConfig): Configurable parameters that are used
            during the training stage.
        model_source (Union[str, Path, AnomalyModule]): Path to the model ckpt file or the Anomaly model.
        meta_data_path (Union[str, Path], optional): Path to metadata file. If none, it tries to load the params
                from the model state_dict. Defaults to None.
    """

    def __init__(
        self,
        config: Union[DictConfig, ListConfig],
        model_source: Union[str, Path, AnomalyModule],
        meta_data_path: Union[str, Path] = None,
    ):
        self.config = config
        if isinstance(model_source, AnomalyModule):
            self.model = model_source
        else:
            self.model = self.load_model(model_source)

        self.meta_data = self._load_meta_data(meta_data_path)

    def _load_meta_data(self, path: Optional[Union[str, Path]] = None) -> Union[Dict, DictConfig]:
        """Load metadata from file or from model state dict.

        Args:
            path (Optional[Union[str, Path]], optional): Path to metadata file. If none, it tries to load the params
                from the model state_dict. Defaults to None.

        Returns:
            Dict: Dictionary containing the meta_data.
        """
        meta_data: Union[DictConfig, Dict[str, Union[float, Tensor, np.ndarray]]]
        if path is None:
            meta_data = get_model_metadata(self.model)
        else:
            meta_data = super()._load_meta_data(path)
        return meta_data

    def load_model(self, path: Union[str, Path]) -> AnomalyModule:
        """Load the PyTorch model.

        Args:
            path (Union[str, Path]): Path to model ckpt file.

        Returns:
            (AnomalyModule): PyTorch Lightning model.
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
                Defaults to None.

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

        if isinstance(anomaly_map, Tensor):
            anomaly_map = anomaly_map.cpu().numpy()

        if "image_shape" in meta_data and anomaly_map.shape != meta_data["image_shape"]:
            anomaly_map = cv2.resize(anomaly_map, meta_data["image_shape"])

        return anomaly_map, float(pred_score)
