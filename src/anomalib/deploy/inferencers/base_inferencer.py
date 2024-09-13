"""Base Inferencer for Torch and OpenVINO."""

# Copyright (C) 2022-2024 Intel Corporation
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
    """Abstract class for the inference.

    This is used by both Torch and OpenVINO inference.
    """

    @abstractmethod
    def load_model(self, path: str | Path) -> Any:  # noqa: ANN401
        """Load Model."""
        raise NotImplementedError

    @abstractmethod
    def pre_process(self, image: np.ndarray) -> np.ndarray | torch.Tensor:
        """Pre-process."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, image: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Forward-Pass input to model."""
        raise NotImplementedError

    @abstractmethod
    def post_process(self, predictions: np.ndarray | torch.Tensor, metadata: dict[str, Any] | None) -> dict[str, Any]:
        """Post-Process."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: str | Path | np.ndarray | torch.Tensor) -> ImageResult:
        """Predict."""
        raise NotImplementedError

    @staticmethod
    def _superimpose_segmentation_mask(metadata: dict, anomaly_map: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Superimpose segmentation mask on top of image.

        Args:
            metadata (dict): Metadata of the image which contains the image size.
            anomaly_map (np.ndarray): Anomaly map which is used to extract segmentation mask.
            image (np.ndarray): Image on which segmentation mask is to be superimposed.

        Returns:
            np.ndarray: Image with segmentation mask superimposed.
        """
        pred_mask = compute_mask(anomaly_map, 0.5)  # assumes predictions are normalized.
        image_height = metadata["image_shape"][0]
        image_width = metadata["image_shape"][1]
        pred_mask = cv2.resize(pred_mask, (image_width, image_height))
        boundaries = find_boundaries(pred_mask)
        outlines = dilation(boundaries, np.ones((7, 7)))
        image[outlines] = [255, 0, 0]
        return image

    def __call__(self, image: np.ndarray) -> ImageResult:
        """Call predict on the Image.

        Args:
            image (np.ndarray): Input Image

        Returns:
            ImageResult: Prediction results to be visualized.
        """
        return self.predict(image)

    @staticmethod
    def _normalize(
        pred_scores: torch.Tensor | np.float32,
        metadata: dict | DictConfig,
        anomaly_maps: torch.Tensor | np.ndarray | None = None,
    ) -> tuple[np.ndarray | torch.Tensor | None, float]:
        """Apply normalization and resizes the image.

        Args:
            pred_scores (Tensor | np.float32): Predicted anomaly score
            metadata (dict | DictConfig): Meta data. Post-processing step sometimes requires
                additional meta data such as image shape. This variable comprises such info.
            anomaly_maps (Tensor | np.ndarray | None): Predicted raw anomaly map.

        Returns:
            tuple[np.ndarray | torch.Tensor | None, float]: Post processed predictions that are ready to be
                visualized and predicted scores.
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

    def _load_metadata(self, path: str | Path | dict | None = None) -> dict | DictConfig:  # noqa: PLR6301
        """Load the meta data from the given path.

        Args:
            path (str | Path | dict | None, optional): Path to JSON file containing the metadata.
                If no path is provided, it returns an empty dict. Defaults to None.

        Returns:
            dict | DictConfig: Dictionary containing the metadata.
        """
        metadata: dict[str, float | np.ndarray | torch.Tensor] | DictConfig = {}
        if path is not None:
            config = OmegaConf.load(path)
            metadata = cast(DictConfig, config)
        return metadata
