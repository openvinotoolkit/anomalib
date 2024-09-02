"""OpenVINO Inferencer implementation."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from omegaconf import DictConfig
from PIL import Image

from anomalib import TaskType
from anomalib.data.utils.label import LabelName
from anomalib.utils.visualization import ImageResult

from .base_inferencer import Inferencer

logger = logging.getLogger("anomalib")

if find_spec("openvino") is not None:
    import openvino as ov

    if TYPE_CHECKING:
        from openvino import CompiledModel
else:
    logger.warning("OpenVINO is not installed. Please install OpenVINO to use OpenVINOInferencer.")


class OpenVINOInferencer(Inferencer):
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
        metadata: str | Path | dict | None = None,
        device: str | None = "AUTO",
        task: str | None = None,
        config: dict | None = None,
    ) -> None:
        self.device = device

        self.config = config
        self.input_blob, self.output_blob, self.model = self.load_model(path)
        self.metadata = super()._load_metadata(metadata)

        self.task = TaskType(task) if task else TaskType(self.metadata["task"])

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

        compile_model = core.compile_model(model=model, device_name=self.device, config=self.config)

        input_blob = compile_model.input(0)
        output_blob = compile_model.output(0)

        return input_blob, output_blob, compile_model

    @staticmethod
    def pre_process(image: np.ndarray) -> np.ndarray:
        """Pre-process the input image by applying transformations.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: pre-processed image.
        """
        processed_image = image

        if len(processed_image.shape) == 3:
            processed_image = np.expand_dims(processed_image, axis=0)

        if processed_image.shape[-1] == 3:
            processed_image = processed_image.transpose(0, 3, 1, 2)

        return processed_image

    def predict(
        self,
        image: str | Path | np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> ImageResult:
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
            image = Image.open(image)

        # Convert PIL image to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image, dtype=np.float32)
        if not isinstance(image, np.ndarray):
            msg = f"Input image must be a numpy array or a path to an image. Got {type(image)}"
            raise TypeError(msg)

        # Resize image to model input size if not dynamic
        if self.input_blob.partial_shape[2].is_static and self.input_blob.partial_shape[3].is_static:
            image = cv2.resize(image, tuple(list(self.input_blob.shape)[2:][::-1]))

        # Normalize numpy array to range [0, 1]
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0

        # Check if metadata is provided, if not use the default metadata.
        if metadata is None:
            metadata = self.metadata if hasattr(self, "metadata") else {}
        metadata["image_shape"] = image.shape[:2]

        processed_image = self.pre_process(image)
        predictions = self.forward(processed_image)
        output = self.post_process(predictions, metadata=metadata)

        return ImageResult(
            image=(image * 255).astype(np.uint8),
            pred_score=output["pred_score"],
            pred_label=output["pred_label"],
            anomaly_map=output["anomaly_map"],
            pred_mask=output["pred_mask"],
            pred_boxes=output["pred_boxes"],
            box_labels=output["box_labels"],
        )

    def forward(self, image: np.ndarray) -> np.ndarray:
        """Forward-Pass input tensor to the model.

        Args:
            image (np.ndarray): Input tensor.

        Returns:
            np.ndarray: Output predictions.
        """
        return self.model(image)

    def post_process(self, predictions: np.ndarray, metadata: dict | DictConfig | None = None) -> dict[str, Any]:
        """Post process the output predictions.

        Args:
            predictions (np.ndarray): Raw output predicted by the model.
            metadata (Dict, optional): Metadata. Post-processing step sometimes requires
                additional metadata such as image shape. This variable comprises such info.
                Defaults to None.

        Returns:
            dict[str, Any]: Post processed prediction results.
        """
        if metadata is None:
            metadata = self.metadata

        predictions = predictions[self.output_blob]

        # Initialize the result variables.
        anomaly_map: np.ndarray | None = None
        pred_label: LabelName | None = None
        pred_mask: float | None = None

        # If predictions returns a single value, this means that the task is
        # classification, and the value is the classification prediction score.
        if len(predictions.shape) == 1:
            task = TaskType.CLASSIFICATION
            pred_score = predictions
        else:
            task = TaskType.SEGMENTATION
            anomaly_map = predictions.squeeze()
            pred_score = anomaly_map.reshape(-1).max()

        # Common practice in anomaly detection is to assign anomalous
        # label to the prediction if the prediction score is greater
        # than the image threshold.
        if "image_threshold" in metadata:
            pred_idx = pred_score >= metadata["image_threshold"]
            pred_label = LabelName.ABNORMAL if pred_idx else LabelName.NORMAL

        if task == TaskType.CLASSIFICATION:
            _, pred_score = self._normalize(pred_scores=pred_score, metadata=metadata)
        elif task in {TaskType.SEGMENTATION, TaskType.DETECTION}:
            if "pixel_threshold" in metadata:
                pred_mask = (anomaly_map >= metadata["pixel_threshold"]).astype(np.uint8)

            anomaly_map, pred_score = self._normalize(
                pred_scores=pred_score,
                anomaly_maps=anomaly_map,
                metadata=metadata,
            )
            if anomaly_map is None:
                msg = "Anomaly map cannot be None."
                raise ValueError(msg)

            if "image_shape" in metadata and anomaly_map.shape != metadata["image_shape"]:
                image_height = metadata["image_shape"][0]
                image_width = metadata["image_shape"][1]
                anomaly_map = cv2.resize(anomaly_map, (image_width, image_height))

                if pred_mask is not None:
                    pred_mask = cv2.resize(pred_mask, (image_width, image_height))
        else:
            msg = f"Unknown task type: {task}"
            raise ValueError(msg)

        if self.task == TaskType.DETECTION:
            pred_boxes = self._get_boxes(pred_mask)
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

    @staticmethod
    def _get_boxes(mask: np.ndarray) -> np.ndarray:
        """Get bounding boxes from masks.

        Args:
            mask (np.ndarray): Input mask of shape (H, W)

        Returns:
            np.ndarray: array of shape (N, 4) containing the bounding box coordinates of the objects in the masks
            in xyxy format.
        """
        _, comps = cv2.connectedComponents(mask)

        labels = np.unique(comps)
        boxes = []
        for label in labels[labels != 0]:
            y_loc, x_loc = np.where(comps == label)
            boxes.append([np.min(x_loc), np.min(y_loc), np.max(x_loc), np.max(y_loc)])
        return np.stack(boxes) if boxes else np.empty((0, 4))
