"""This module contains Torch inference implementations."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor, nn
from torchvision.transforms.functional import resize

from anomalib.data.utils import read_image
from anomalib.post_processing import ImageResult

LABEL_MAPPING = {0: "normal", 1: "anomaly"}


class TorchInferencer:
    """PyTorch implementation for the inference.

    Args:
        path (str | Path): Path to Torch model weights.
        device (str): Device to use for inference. Options are auto, cpu, cuda. Defaults to "auto".
        task (str): Task type. Defaults to "classification".
    """

    def __init__(self, path: str | Path, device: str = "auto", task: str = "classification") -> None:
        self.device = self._get_device(device)

        # Load the model weights.
        self.model = self.load_model(path)
        self.input_size = torch.load(
            path,
        )["input_size"]
        self.task = task

    @staticmethod
    def _get_device(device: str) -> torch.device:
        """Get the device to use for inference.

        Args:
            device (str): Device to use for inference. Options are auto, cpu, cuda.

        Returns:
            torch.device: Device to use for inference.
        """
        if device not in ("auto", "cpu", "cuda", "gpu"):
            raise ValueError(f"Unknown device {device}")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "gpu":
            device = "cuda"
        return torch.device(device)

    def load_model(self, path: str | Path) -> nn.Module:
        """Load the PyTorch model.

        Args:
            path (str | Path): Path to model ckpt file.

        Returns:
            (AnomalyModule): PyTorch Lightning model.
        """

        model = torch.load(path, map_location=self.device)["model"]
        model.eval()
        return model.to(self.device)

    def pre_process(self, image: np.ndarray) -> Tensor:
        """Pre process the input image by applying transformations.

        Args:
            image (np.ndarray): Input image

        Returns:
            Tensor: pre-processed image.
        """
        processed_image = torch.from_numpy(image).to(self.device)

        if len(processed_image.shape) == 3:
            processed_image = processed_image.permute(2, 0, 1)
            processed_image = processed_image.unsqueeze(0)
            processed_image = resize(processed_image, size=self.input_size)

        return processed_image.to(self.device)

    def forward(self, image: Tensor) -> Tensor:
        """Forward-Pass input tensor to the model.

        Args:
            image (Tensor): Input tensor.

        Returns:
            Tensor: Output predictions.
        """
        return self.model(image)

    def post_process(self, predictions: dict[str, Any], image_shape: tuple[int, int]) -> dict[str, Any]:
        """Post process the output predictions.

        Args:
            predictions (Tensor): Raw output predicted by the model.
            image_shape (tuple[int, int]): Shape of the input image.

        Returns:
            dict[str, str | float | np.ndarray]: Post processed prediction results.
        """
        for key, value in predictions.items():
            if isinstance(value, Tensor):
                predictions[key] = value.detach().cpu().numpy()

        # reshape output to original image size
        for key, value in predictions.items():
            if key in ("anomaly_map", "pred_mask") and value is not None:
                predictions[key] = cv2.resize(value, dsize=image_shape[::-1])

        if predictions["pred_label"] is not None:
            predictions["pred_label"] = LABEL_MAPPING[predictions["pred_label"].item()]

        return predictions

    def predict(
        self,
        image: str | Path | np.ndarray,
    ) -> ImageResult:
        """Perform a prediction for a given input image.

        The main workflow is (i) pre-processing, (ii) forward-pass, (iii) post-process.

        Args:
            image (Union[str, np.ndarray]): Input image whose output is to be predicted.
                It could be either a path to image or numpy array itself.

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
