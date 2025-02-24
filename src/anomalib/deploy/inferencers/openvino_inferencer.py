"""OpenVINO Inferencer for optimized model inference.

This module provides the OpenVINO inferencer implementation for running optimized
inference with OpenVINO IR models.

Example:
    Assume we have OpenVINO IR model files in the following structure:

    .. code-block:: bash

        $ tree weights
        ./weights
        ├── model.bin
        ├── model.xml
        └── metadata.json

    Create an OpenVINO inferencer:

    >>> from anomalib.deploy import OpenVINOInferencer
    >>> inferencer = OpenVINOInferencer(
    ...     path="weights/model.xml",
    ...     device="CPU"
    ... )

    Make predictions:

    >>> # From image path
    >>> prediction = inferencer.predict("path/to/image.jpg")

    >>> # From PIL Image
    >>> from PIL import Image
    >>> image = Image.open("path/to/image.jpg")
    >>> prediction = inferencer.predict(image)

    >>> # From numpy array
    >>> import numpy as np
    >>> image = np.random.rand(224, 224, 3)
    >>> prediction = inferencer.predict(image)

    The prediction result contains anomaly maps and scores:

    >>> prediction.anomaly_map  # doctest: +SKIP
    array([[0.1, 0.2, ...]], dtype=float32)

    >>> prediction.pred_score  # doctest: +SKIP
    0.86
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning_utilities.core.imports import module_available
from openvino.runtime.utils.data_helpers.wrappers import OVDict
from PIL.Image import Image as PILImage

from anomalib.data import NumpyImageBatch
from anomalib.data.utils import read_image

logger = logging.getLogger("anomalib")


class OpenVINOInferencer:
    """OpenVINO inferencer for optimized model inference.

    Args:
        path (str | Path | tuple[bytes, bytes]): Path to OpenVINO IR files
            (``.xml`` and ``.bin``) or ONNX model, or tuple of xml/bin data as
            bytes.
        device (str | None, optional): Inference device.
            Options: ``"AUTO"``, ``"CPU"``, ``"GPU"``, ``"NPU"``.
            Defaults to ``"AUTO"``.
        config (dict | None, optional): OpenVINO configuration parameters.
            Defaults to ``None``.

    Example:
        >>> from anomalib.deploy import OpenVINOInferencer
        >>> model = OpenVINOInferencer(
        ...     path="model.xml",
        ...     device="CPU"
        ... )
        >>> prediction = model.predict("test.jpg")
    """

    def __init__(
        self,
        path: str | Path | tuple[bytes, bytes],
        device: str | None = "AUTO",
        config: dict | None = None,
    ) -> None:
        if not module_available("openvino"):
            msg = "OpenVINO is not installed. Please install OpenVINO to use OpenVINOInferencer."
            raise ImportError(msg)

        self.device = device
        self.config = config
        self.input_blob, self.output_blob, self.model = self.load_model(path)

    def load_model(self, path: str | Path | tuple[bytes, bytes]) -> tuple[Any, Any, Any]:
        """Load OpenVINO model from file or bytes.

        Args:
            path (str | Path | tuple[bytes, bytes]): Path to model files or model
                data as bytes tuple.

        Returns:
            tuple[Any, Any, Any]: Tuple containing:
                - Input blob
                - Output blob
                - Compiled model

        Raises:
            ValueError: If model path has invalid extension.
        """
        import openvino as ov

        core = ov.Core()
        # If tuple of bytes is passed
        if isinstance(path, tuple):
            model = core.read_model(model=path[0], weights=path[1])
        else:
            path = path if isinstance(path, Path) else Path(path)
            if path.suffix in {".bin", ".xml"}:
                if path.suffix == ".bin":
                    bin_path, xml_path = path, path.with_suffix(".xml")
                elif path.suffix == ".xml":
                    xml_path, bin_path = path, path.with_suffix(".bin")
                model = core.read_model(xml_path, bin_path)
            elif path.suffix == ".onnx":
                model = core.read_model(path)
            else:
                msg = f"Path must be .onnx, .bin or .xml file. Got {path.suffix}"
                raise ValueError(msg)
        # Create cache folder
        cache_folder = Path("cache")
        cache_folder.mkdir(exist_ok=True)
        core.set_property({"CACHE_DIR": cache_folder})

        compile_model = core.compile_model(
            model=model,
            device_name=self.device,
            config=self.config,
        )

        input_blob = compile_model.input(0)
        output_blob = compile_model.output(0)

        return input_blob, output_blob, compile_model

    @staticmethod
    def pre_process(image: np.ndarray) -> np.ndarray:
        """Pre-process input image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Pre-processed image with shape (N,C,H,W).
        """
        # Normalize numpy array to range [0, 1]
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0

        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        if image.shape[-1] == 3:
            image = image.transpose(0, 3, 1, 2)

        return image

    @staticmethod
    def post_process(predictions: OVDict) -> dict:
        """Convert OpenVINO predictions to dictionary.

        Args:
            predictions (OVDict): Raw predictions from OpenVINO model.

        Returns:
            dict: Dictionary of prediction tensors.
        """
        names = [next(iter(name)) for name in predictions.names()]
        values = predictions.to_tuple()
        return dict(zip(names, values, strict=False))

    def predict(self, image: str | Path | np.ndarray | PILImage | torch.Tensor) -> NumpyImageBatch:
        """Run inference on an input image.

        Args:
            image (str | Path | np.ndarray | PILImage | torch.Tensor): Input image as file path or array.

        Returns:
            NumpyImageBatch: Batch containing the predictions.

        Raises:
            TypeError: If image input is invalid type.
        """
        # Convert file path or string to image if necessary
        if isinstance(image, str | Path):
            image = read_image(image, as_tensor=False)
        elif isinstance(image, PILImage):
            image = np.array(image) / 255.0
        elif isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        image = self.pre_process(image)
        predictions = self.model(image)
        pred_dict = self.post_process(predictions)

        return NumpyImageBatch(image=image, **pred_dict)
