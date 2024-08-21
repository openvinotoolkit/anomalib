"""Torch inference implementations."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from torch import nn

from anomalib.data.utils import read_image
from anomalib.dataclasses import ImageBatch


class TorchInferencer:
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

        >>> from anomalib.data.utils import read_image
        >>> image = read_image("path/to/image.jpg")
        >>> result = inferencer.predict(image)

        ``result`` will be an ``PredictBatch`` object containing the prediction
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

        # Load the model weights and metadata
        self.model = self.load_model(path)

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

    def predict(
        self,
        image: str | Path | torch.Tensor,
    ) -> ImageBatch:
        """Perform a prediction for a given input image.

        Args:
            image (Union[str, np.ndarray]): Input image whose output is to be predicted.
                It could be either a path to image or the tensor itself.

        Returns:
            ImageResult: Prediction results to be visualized.
        """
        if isinstance(image, str | Path):
            image = read_image(image, as_tensor=True)

        image = self.pre_process(image)
        predictions = self.model(image)

        return ImageBatch(
            image=image,
            **predictions._asdict(),
        )

    def pre_process(self, image: torch.Tensor) -> torch.Tensor:
        """Pre process the input image.

        Args:
            image (torch.Tensor): Input image

        Returns:
            Tensor: pre-processed image.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)  # model expects [B, C, H, W]

        return image.to(self.device)
