"""Base Inferencer for Torch and OpenVINO.

This module provides the base inferencer class that defines the interface for
performing inference with anomaly detection models.

The base class is used by both the PyTorch and OpenVINO inferencers to ensure
a consistent API across different backends.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from skimage.morphology import dilation
from skimage.segmentation import find_boundaries

from anomalib.utils.normalization.min_max import normalize as normalize_min_max
from anomalib.utils.post_processing import compute_mask
from anomalib.utils.visualization import ImageResult


class Inferencer(ABC):
    """Abstract base class for performing inference with anomaly detection models.

    This class defines the interface that must be implemented by concrete
    inferencer classes for different backends (PyTorch, OpenVINO).

    Example:
        >>> from anomalib.deploy import TorchInferencer
        >>> model = TorchInferencer(path="path/to/model.pt")
        >>> predictions = model.predict(image="path/to/image.jpg")
    """

    @abstractmethod
    def load_model(self, path: str | Path) -> Any:  # noqa: ANN401
        """Load a model from the specified path.

        Args:
            path (str | Path): Path to the model file.

        Returns:
            Any: Loaded model instance.

        Raises:
            NotImplementedError: This is an abstract method.
        """
        raise NotImplementedError

    @abstractmethod
    def pre_process(self, image: np.ndarray) -> np.ndarray | torch.Tensor:
        """Pre-process an input image.

        Args:
            image (np.ndarray): Input image to pre-process.

        Returns:
            np.ndarray | torch.Tensor: Pre-processed image.

        Raises:
            NotImplementedError: This is an abstract method.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, image: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Perform a forward pass on the model.

        Args:
            image (np.ndarray | torch.Tensor): Pre-processed input image.

        Returns:
            np.ndarray | torch.Tensor: Model predictions.

        Raises:
            NotImplementedError: This is an abstract method.
        """
        raise NotImplementedError

    @abstractmethod
    def post_process(self, predictions: np.ndarray | torch.Tensor, metadata: dict[str, Any] | None) -> dict[str, Any]:
        """Post-process model predictions.

        Args:
            predictions (np.ndarray | torch.Tensor): Raw model predictions.
            metadata (dict[str, Any] | None): Metadata used for post-processing.

        Returns:
            dict[str, Any]: Post-processed predictions.

        Raises:
            NotImplementedError: This is an abstract method.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: str | Path | np.ndarray | torch.Tensor) -> ImageResult:
        """Run inference on an image.

        Args:
            image (str | Path | np.ndarray | torch.Tensor): Input image.

        Returns:
            ImageResult: Prediction results.

        Raises:
            NotImplementedError: This is an abstract method.
        """
        raise NotImplementedError

    @staticmethod
    def _superimpose_segmentation_mask(metadata: dict, anomaly_map: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Superimpose segmentation mask on an image.

        Args:
            metadata (dict): Image metadata containing the image dimensions.
            anomaly_map (np.ndarray): Anomaly map used to extract segmentation.
            image (np.ndarray): Image on which to superimpose the mask.

        Returns:
            np.ndarray: Image with superimposed segmentation mask.

        Example:
            >>> image = np.zeros((100, 100, 3))
            >>> anomaly_map = np.zeros((100, 100))
            >>> metadata = {"image_shape": (100, 100)}
            >>> result = Inferencer._superimpose_segmentation_mask(
            ...     metadata,
            ...     anomaly_map,
            ...     image,
            ... )
            >>> result.shape
            (100, 100, 3)
        """
        pred_mask = compute_mask(anomaly_map, 0.5)  # assumes normalized preds
        image_height = metadata["image_shape"][0]
        image_width = metadata["image_shape"][1]
        pred_mask = cv2.resize(pred_mask, (image_width, image_height))
        boundaries = find_boundaries(pred_mask)
        outlines = dilation(boundaries, np.ones((7, 7)))
        image[outlines] = [255, 0, 0]
        return image

    def __call__(self, image: np.ndarray) -> ImageResult:
        """Call predict on an image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            ImageResult: Prediction results to be visualized.

        Example:
            >>> model = Inferencer()  # doctest: +SKIP
            >>> image = np.zeros((100, 100, 3))
            >>> predictions = model(image)  # doctest: +SKIP
        """
        return self.predict(image)

    @staticmethod
    def _normalize(
        pred_scores: torch.Tensor | np.float32,
        metadata: dict | DictConfig,
        anomaly_maps: torch.Tensor | np.ndarray | None = None,
    ) -> tuple[np.ndarray | torch.Tensor | None, float]:
        """Normalize predictions using min-max normalization.

        Args:
            pred_scores (torch.Tensor | np.float32): Predicted anomaly scores.
            metadata (dict | DictConfig): Metadata containing normalization
                parameters.
            anomaly_maps (torch.Tensor | np.ndarray | None): Raw anomaly maps.
                Defaults to None.

        Returns:
            tuple[np.ndarray | torch.Tensor | None, float]: Normalized predictions
                and scores.

        Example:
            >>> scores = torch.tensor(0.5)
            >>> metadata = {
            ...     "image_threshold": 0.5,
            ...     "pred_scores.min": 0.0,
            ...     "pred_scores.max": 1.0
            ... }
            >>> maps, norm_scores = Inferencer._normalize(scores, metadata)
            >>> norm_scores
            0.5
        """
        # min max normalization
        if "pred_scores.min" in metadata and "pred_scores.max" in metadata:
            if anomaly_maps is not None and "anomaly_maps.max" in metadata:
                anomaly_maps = normalize_min_max(
                    anomaly_maps,
                    metadata["pixel_threshold"],
                    metadata["anomaly_maps.min"],
                    metadata["anomaly_maps.max"],
                )
            pred_scores = normalize_min_max(
                pred_scores,
                metadata["image_threshold"],
                metadata["pred_scores.min"],
                metadata["pred_scores.max"],
            )

        return anomaly_maps, float(pred_scores)

    @staticmethod
    def _load_metadata(path: str | Path | dict | None = None) -> dict | DictConfig:
        """Load metadata from a file.

        Args:
            path (str | Path | dict | None): Path to metadata file. If None,
                returns empty dict. Defaults to None.

        Returns:
            dict | DictConfig: Loaded metadata.

        Example:
            >>> model = Inferencer()  # doctest: +SKIP
            >>> metadata = model._load_metadata("path/to/metadata.json")
            ... # doctest: +SKIP
        """
        metadata: dict[str, float | np.ndarray | torch.Tensor] | DictConfig = {}
        if path is not None:
            config = OmegaConf.load(path)
            metadata = cast(DictConfig, config)
        return metadata
