"""Torch inference implementations."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Sequence
from pathlib import Path
from typing import Any

import albumentations as A  # noqa: N812
import cv2
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from anomalib import TaskType
from anomalib.data.utils.boxes import masks_to_boxes

from .base_inferencer import Inferencer


class TorchInferencer(Inferencer):
    """PyTorch implementation for the inference.

    Args:
        path (str | Path): Path to Torch model weights.
        device (str): Device to use for inference. Options are ``auto``,
            ``cpu``, ``cuda``.
            Defaults to ``auto``.

    Examples:
        Assume that we have a Torch ``pt`` model and metadata files in the
        following structure:

        >>> from anomalib.deploy.inferencers import TorchInferencer
        >>> inferencer = TorchInferencer(path="path/to/torch/model.pt", device="cpu")

        This will ensure that the model is loaded on the ``CPU`` device. To make
        a prediction, we can simply call the ``predict`` method:

        >>> import cv2
        >>> image = cv2.imread("path/to/image.jpg")
        >>> image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        >>> result = inferencer.predict(image)

        ``result`` will be an ``ImageResult`` object containing the prediction
        results. For example, to visualize the heatmap, we can do the following:

        >>> from matplotlib import pyplot as plt
        >>> plt.imshow(result.heatmap)

        It is also possible to visualize the true and predicted masks if the
        task is ``TaskType.SEGMENTATION``:

        >>> plt.imshow(result.gt_mask)
        >>> plt.imshow(result.pred_mask)
    """

    def __init__(
        self,
        path: str | Path,
        device: str = "auto",
    ) -> None:
        self.device = self._get_device(device)

        # Load the model weights, metadata and data transforms.
        self.checkpoint = self._load_checkpoint(path)
        self.model = self.load_model(path)
        self.metadata = self._load_metadata(path)
        self.transform = A.from_dict(self.metadata["transform"])

    @staticmethod
    def _get_device(device: str) -> torch.device:
        """Get the device to use for inference.

        Args:
            device (str): Device to use for inference. Options are auto, cpu, cuda.

        Returns:
            torch.device: Device to use for inference.
        """
        if device not in ("auto", "cpu", "cuda", "gpu"):
            msg = f"Unknown device {device}"
            raise ValueError(msg)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "gpu":
            device = "cuda"
        return torch.device(device)

    def _load_checkpoint(self, path: str | Path) -> dict:
        """Load the checkpoint.

        Args:
            path (str | Path): Path to the torch ckpt file.

        Returns:
            dict: Dictionary containing the model and metadata.
        """
        if isinstance(path, str):
            path = Path(path)

        if path.suffix not in (".pt", ".pth"):
            msg = f"Unknown torch checkpoint file format {path.suffix}. Make sure you save the Torch model."
            raise ValueError(msg)

        return torch.load(path, map_location=self.device)

    def _load_metadata(self, path: str | Path | dict | None = None) -> dict | DictConfig:
        """Load metadata from file.

        Args:
            path (str | Path | dict): Path to the model pt file.

        Returns:
            dict: Dictionary containing the metadata.
        """
        metadata: dict | DictConfig

        if isinstance(path, dict):
            metadata = path
        elif isinstance(path, str | Path):
            checkpoint = self._load_checkpoint(path)

            # Torch model should ideally contain the metadata in the checkpoint.
            # Check if the metadata is present in the checkpoint.
            if "metadata" not in checkpoint:
                msg = (
                    "``metadata`` is not found in the checkpoint. Please ensure that you save the model as Torch model."
                )
                raise KeyError(
                    msg,
                )
            metadata = checkpoint["metadata"]
        else:
            msg = f"Unknown ``path`` type {type(path)}"
            raise TypeError(msg)

        return metadata

    def load_model(self, path: str | Path) -> nn.Module:
        """Load the PyTorch model.

        Args:
            path (str | Path): Path to the Torch model.

        Returns:
            (nn.Module): Torch model.
        """
        checkpoint = self._load_checkpoint(path)
        if "model" not in checkpoint:
            msg = "``model`` is not found in the checkpoint. Please check the checkpoint file."
            raise KeyError(msg)

        model = checkpoint["model"]
        model.eval()
        return model.to(self.device)

    def pre_process(self, image: np.ndarray) -> torch.Tensor:
        """Pre process the input image by applying transformations.

        Args:
            image (np.ndarray): Input image

        Returns:
            Tensor: pre-processed image.
        """
        processed_image = self.transform(image=image)["image"]

        if len(processed_image) == 3:
            processed_image = processed_image.unsqueeze(0)

        return processed_image.to(self.device)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward-Pass input tensor to the model.

        Args:
            image (torch.Tensor): Input tensor.

        Returns:
            Tensor: Output predictions.
        """
        return self.model(image)

    def post_process(
        self,
        predictions: torch.Tensor | list[torch.Tensor] | dict[str, torch.Tensor],
        metadata: dict | DictConfig | None = None,
    ) -> dict[str, Any]:
        """Post process the output predictions.

        Args:
            predictions (Tensor | list[torch.Tensor] | dict[str, torch.Tensor]): Raw output predicted by the model.
            metadata (dict, optional): Meta data. Post-processing step sometimes requires
                additional meta data such as image shape. This variable comprises such info.
                Defaults to None.

        Returns:
            dict[str, str | float | np.ndarray]: Post processed prediction results.
        """
        if metadata is None:
            metadata = self.metadata

        # Some models return a Tensor while others return a list or dictionary. Handle both cases.
        # TODO(ashwinvaidya17): Wrap this post-processing stage within the model's forward pass.
        # CVS-122674

        # Case I: Predictions could be a tensor.
        if isinstance(predictions, torch.Tensor):
            anomaly_map = predictions.detach().cpu().numpy()
            pred_score = anomaly_map.reshape(-1).max()

        # Case II: Predictions could be a dictionary of tensors.
        elif isinstance(predictions, dict):
            if "anomaly_map" in predictions:
                anomaly_map = predictions["anomaly_map"].detach().cpu().numpy()
            else:
                msg = "``anomaly_map`` not found in the predictions."
                raise KeyError(msg)

            if "pred_score" in predictions:
                pred_score = predictions["pred_score"].detach().cpu().numpy()
            else:
                pred_score = anomaly_map.reshape(-1).max()

        # Case III: Predictions could be a list of tensors.
        elif isinstance(predictions, Sequence):
            if isinstance(predictions[1], (torch.Tensor)):
                anomaly_map, pred_score = predictions
                anomaly_map = anomaly_map.detach().cpu().numpy()
                pred_score = pred_score.detach().cpu().numpy()
            else:
                anomaly_map, pred_score = predictions
                pred_score = pred_score.detach()
        else:
            msg = (
                f"Unknown prediction type {type(predictions)}. "
                "Expected torch.Tensor, list[torch.Tensor] or dict[str, torch.Tensor]."
            )
            raise TypeError(msg)

        # Common practice in anomaly detection is to assign anomalous
        # label to the prediction if the prediction score is greater
        # than the image threshold.
        pred_label: str | None = None
        if "image_threshold" in metadata:
            pred_idx = pred_score >= metadata["image_threshold"]
            pred_label = "Anomalous" if pred_idx else "Normal"

        pred_mask: np.ndarray | None = None
        if "pixel_threshold" in metadata:
            pred_mask = (anomaly_map >= metadata["pixel_threshold"]).squeeze().astype(np.uint8)

        anomaly_map = anomaly_map.squeeze()
        anomaly_map, pred_score = self._normalize(anomaly_maps=anomaly_map, pred_scores=pred_score, metadata=metadata)

        if isinstance(anomaly_map, torch.Tensor):
            anomaly_map = anomaly_map.detach().cpu().numpy()

        if "image_shape" in metadata and anomaly_map.shape != metadata["image_shape"]:
            image_height = metadata["image_shape"][0]
            image_width = metadata["image_shape"][1]
            anomaly_map = cv2.resize(anomaly_map, (image_width, image_height))

            if pred_mask is not None:
                pred_mask = cv2.resize(pred_mask, (image_width, image_height))

        if self.metadata["task"] == TaskType.DETECTION:
            pred_boxes = masks_to_boxes(torch.from_numpy(pred_mask))[0][0].numpy()
            box_labels = np.ones(pred_boxes.shape[0])
        else:
            pred_boxes = None
            box_labels = None

        return {
            "anomaly_map": anomaly_map,
            "pred_label": pred_label,
            "pred_score": pred_score,
            "pred_mask": pred_mask,
            "pred_boxes": pred_boxes,
            "box_labels": box_labels,
        }
