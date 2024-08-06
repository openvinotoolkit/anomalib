"""OpenVINO Inferencer implementation."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from openvino.runtime.utils.data_helpers.wrappers import OVDict

from anomalib.data.utils import read_image
from anomalib.dataclasses import NumpyBatch

logger = logging.getLogger("anomalib")


if find_spec("openvino") is not None:
    import openvino as ov

    if TYPE_CHECKING:
        from openvino import CompiledModel
else:
    logger.warning("OpenVINO is not installed. Please install OpenVINO to use OpenVINOInferencer.")


class OpenVINOInferencer:
    """OpenVINO implementation for the inference.

    Args:
        path (str | Path): Path to the openvino onnx, xml or bin file.
        metadata (str | Path | dict, optional): Path to metadata file or a dict object defining the
            metadata.
            Defaults to ``None``.
        device (str | None, optional): Device to run the inference on (AUTO, CPU, GPU, NPU).
            Defaults to ``AUTO``.
        task (TaskType | None, optional): Task type.
            Defaults to ``None``.
        config (dict | None, optional): Configuration parameters for the inference
            Defaults to ``None``.

    Examples:
        Assume that we have an OpenVINO IR model and metadata files in the following structure:

        .. code-block:: bash

            $ tree weights
            ./weights
            ├── model.bin
            ├── model.xml
            └── metadata.json

        We could then create ``OpenVINOInferencer`` as follows:

        >>> from anomalib.deploy.inferencers import OpenVINOInferencer
        >>> inferencer = OpenVINOInferencer(
        ...     path="weights/model.xml",
        ...     metadata="weights/metadata.json",
        ...     device="CPU",
        ... )

        This will ensure that the model is loaded on the ``CPU`` device and the
        metadata is loaded from the ``metadata.json`` file. To make a prediction,
        we can simply call the ``predict`` method:

        >>> prediction = inferencer.predict(image="path/to/image.jpg")

        Alternatively we can also pass the image as a PIL image or numpy array:

        >>> from PIL import Image
        >>> image = Image.open("path/to/image.jpg")
        >>> prediction = inferencer.predict(image=image)

        >>> import numpy as np
        >>> image = np.random.rand(224, 224, 3)
        >>> prediction = inferencer.predict(image=image)

        ``prediction`` will be an ``ImageResult`` object containing the prediction
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
        path: str | Path | tuple[bytes, bytes],
        device: str | None = "AUTO",
        config: dict | None = None,
    ) -> None:
        self.device = device

        self.config = config
        self.input_blob, self.output_blob, self.model = self.load_model(path)

    def load_model(self, path: str | Path | tuple[bytes, bytes]) -> tuple[Any, Any, "CompiledModel"]:
        """Load the OpenVINO model.

        Args:
            path (str | Path | tuple[bytes, bytes]): Path to the onnx or xml and bin files
                                                        or tuple of .xml and .bin data as bytes.

        Returns:
            [tuple[str, str, ExecutableNetwork]]: Input and Output blob names
                together with the Executable network.
        """
        core = ov.Core()
        # If tuple of bytes is passed
        if isinstance(path, tuple):
            model = core.read_model(model=path[0], weights=path[1])
        else:
            path = path if isinstance(path, Path) else Path(path)
            if path.suffix in (".bin", ".xml"):
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

        compile_model = core.compile_model(model=model, device_name=self.device, config=self.config)

        input_blob = compile_model.input(0)
        output_blob = compile_model.output(0)

        return input_blob, output_blob, compile_model

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        """Pre-process the input image by applying transformations.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: pre-processed image.
        """
        # Resize image to model input size if not dynamic
        if self.input_blob.partial_shape[2].is_static and self.input_blob.partial_shape[3].is_static:
            image = cv2.resize(image, tuple(list(self.input_blob.shape)[2:][::-1]))

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
        """Convert OpenVINO output dictionary to NumpyBatch."""
        names = [next(iter(name)) for name in predictions.names()]
        values = predictions.to_tuple()
        return dict(zip(names, values, strict=False))

    def predict(
        self,
        image: str | Path | np.ndarray,
    ) -> NumpyBatch:
        """Perform a prediction for a given input image.

        The main workflow is (i) pre-processing, (ii) forward-pass, (iii) post-process.

        Args:
            image (Union[str, np.ndarray]): Input image whose output is to be predicted.
                It could be either a path to image or numpy array itself.

            metadata: Metadata information such as shape, threshold.

        Returns:
            ImageResult: Prediction results to be visualized.
        """
        # Convert file path or string to image if necessary
        if isinstance(image, str | Path):
            image = read_image(image, as_tensor=False)
        if not isinstance(image, np.ndarray):
            msg = f"Input image must be a numpy array or a path to an image. Got {type(image)}"
            raise TypeError(msg)

        image = self.pre_process(image)
        predictions = self.model(image)
        pred_dict = self.post_process(predictions)

        return NumpyBatch(
            image=image,
            **pred_dict,
        )
