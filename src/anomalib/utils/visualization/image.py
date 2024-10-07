"""Image/video generator."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries

from anomalib import TaskType
from anomalib.data.utils import read_image
from anomalib.utils.post_processing import add_anomalous_label, add_normal_label, draw_boxes, superimpose_anomaly_map

from .base import BaseVisualizer, GeneratorResult, VisualizationStep

if TYPE_CHECKING:
    from matplotlib.axis import Axes


class VisualizationMode(str, Enum):
    """Type of visualization mode."""

    FULL = "full"
    SIMPLE = "simple"


class ImageResult:
    """Collection of data needed to visualize the predictions for an image."""

    def __init__(
        self,
        image: np.ndarray,
        pred_score: float,
        pred_label: str,
        anomaly_map: np.ndarray | None = None,
        gt_mask: np.ndarray | None = None,
        pred_mask: np.ndarray | None = None,
        gt_boxes: np.ndarray | None = None,
        pred_boxes: np.ndarray | None = None,
        box_labels: np.ndarray | None = None,
        normalize: bool = False,
    ) -> None:
        self.anomaly_map = anomaly_map
        self.box_labels = box_labels
        self.gt_boxes = gt_boxes
        self.gt_mask = gt_mask
        self.image = image
        self.pred_score = pred_score
        self.pred_label = pred_label
        self.pred_boxes = pred_boxes
        self.heat_map: np.ndarray | None = None
        self.segmentations: np.ndarray | None = None
        self.normal_boxes: np.ndarray | None = None
        self.anomalous_boxes: np.ndarray | None = None

        if anomaly_map is not None:
            self.heat_map = superimpose_anomaly_map(self.anomaly_map, self.image, normalize=normalize)

        if self.gt_mask is not None and self.gt_mask.max() <= 1.0:
            self.gt_mask *= 255

        self.pred_mask = pred_mask
        if self.pred_mask is not None and self.pred_mask.max() <= 1.0:
            self.pred_mask *= 255
            self.segmentations = mark_boundaries(self.image, self.pred_mask, color=(1, 0, 0), mode="thick")
            if self.segmentations.max() <= 1.0:
                self.segmentations = (self.segmentations * 255).astype(np.uint8)

        if self.pred_boxes is not None:
            if self.box_labels is None:
                msg = "Box labels must be provided when box locations are provided."
                raise ValueError(msg)

            self.normal_boxes = self.pred_boxes[~self.box_labels.astype(bool)]
            self.anomalous_boxes = self.pred_boxes[self.box_labels.astype(bool)]

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        repr_str = (
            f"ImageResult(image={self.image}, pred_score={self.pred_score}, pred_label={self.pred_label}, "
            f"anomaly_map={self.anomaly_map}, gt_mask={self.gt_mask}, "
            f"gt_boxes={self.gt_boxes}, pred_boxes={self.pred_boxes}, box_labels={self.box_labels}"
        )
        repr_str += f", pred_mask={self.pred_mask}" if self.pred_mask is not None else ""
        repr_str += f", heat_map={self.heat_map}" if self.heat_map is not None else ""
        repr_str += f", segmentations={self.segmentations}" if self.segmentations is not None else ""
        repr_str += f", normal_boxes={self.normal_boxes}" if self.normal_boxes is not None else ""
        repr_str += f", anomalous_boxes={self.anomalous_boxes}" if self.anomalous_boxes is not None else ""
        repr_str += ")"
        return repr_str


class ImageVisualizer(BaseVisualizer):
    """Image/video generator.

    Args:
        mode (VisualizationMode, optional): Type of visualization mode. Defaults to VisualizationMode.FULL.
        task (TaskType, optional): Type of task. Defaults to TaskType.CLASSIFICATION.
        normalize (bool, optional): Whether or not the anomaly maps should be normalized to image min-max at image
            level. Defaults to False. Note: This is more useful when NormalizationMethod is set to None. Otherwise,
            the overlayed anomaly map will contain the raw scores.
    """

    def __init__(
        self,
        mode: VisualizationMode = VisualizationMode.FULL,
        task: TaskType | str = TaskType.CLASSIFICATION,
        normalize: bool = False,
    ) -> None:
        super().__init__(VisualizationStep.BATCH)
        self.mode = mode
        self.task = task
        self.normalize = normalize

    def generate(self, **kwargs) -> Iterator[GeneratorResult]:
        """Generate images and return them as an iterator."""
        outputs = kwargs.get("outputs", None)
        if outputs is None:
            msg = "Outputs must be provided to generate images."
            raise ValueError(msg)
        return self._visualize_batch(outputs)

    def _visualize_batch(self, batch: dict) -> Iterator[GeneratorResult]:
        """Yield a visualization result for each item in the batch.

        Args:
            batch (dict): Dictionary containing the ground truth and predictions of a batch of images.

        Returns:
            Generator that yields a display-ready visualization for each image.
        """
        batch_size = batch["image"].shape[0]
        for i in range(batch_size):
            if "image_path" in batch:
                height, width = batch["image"].shape[-2:]
                image = (read_image(path=batch["image_path"][i]) * 255).astype(np.uint8)
                image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_AREA)
            elif "video_path" in batch:
                height, width = batch["image"].shape[-2:]
                image = batch["original_image"][i].squeeze().cpu().numpy()
                image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_AREA)
            else:
                msg = "Batch must have either 'image_path' or 'video_path' defined."
                raise KeyError(msg)

            file_name = None
            if "image_path" in batch:
                file_name = Path(batch["image_path"][i])
            elif "video_path" in batch:
                zero_fill = int(np.log10(batch["last_frame"][i])) + 1
                suffix = f"{str(batch['frames'][i].int().item()).zfill(zero_fill)}.png"
                file_name = Path(batch["video_path"][i]) / suffix

            image_result = ImageResult(
                image=image,
                pred_score=batch["pred_scores"][i].cpu().numpy().item() if "pred_scores" in batch else None,
                pred_label=batch["pred_labels"][i].cpu().numpy().item() if "pred_labels" in batch else None,
                anomaly_map=batch["anomaly_maps"][i].cpu().numpy() if "anomaly_maps" in batch else None,
                pred_mask=batch["pred_masks"][i].squeeze().int().cpu().numpy() if "pred_masks" in batch else None,
                gt_mask=batch["mask"][i].squeeze().int().cpu().numpy() if "mask" in batch else None,
                gt_boxes=batch["boxes"][i].cpu().numpy() if "boxes" in batch else None,
                pred_boxes=batch["pred_boxes"][i].cpu().numpy() if "pred_boxes" in batch else None,
                box_labels=batch["box_labels"][i].cpu().numpy() if "box_labels" in batch else None,
                normalize=self.normalize,
            )
            yield GeneratorResult(image=self.visualize_image(image_result), file_name=file_name)

    def visualize_image(self, image_result: ImageResult) -> np.ndarray:
        """Generate the visualization for an image.

        Args:
            image_result (ImageResult): GT and Prediction data for a single image.

        Returns:
            The full or simple visualization for the image, depending on the specified mode.
        """
        if self.mode == VisualizationMode.FULL:
            return self._visualize_full(image_result)
        if self.mode == VisualizationMode.SIMPLE:
            return self._visualize_simple(image_result)
        msg = f"Unknown visualization mode: {self.mode}"
        raise ValueError(msg)

    def _visualize_full(self, image_result: ImageResult) -> np.ndarray:
        """Generate the full set of visualization for an image.

        The full visualization mode shows a grid with subplots that contain the original image, the GT mask (if
        available), the predicted heat map, the predicted segmentation mask (if available), and the predicted
        segmentations (if available).

        Args:
            image_result (ImageResult): GT and Prediction data for a single image.

        Returns:
            An image showing the full set of visualizations for the input image.
        """
        image_grid = _ImageGrid()
        if self.task == TaskType.DETECTION:
            if image_result.pred_boxes is None:
                msg = "Image result predicted boxes are None."
                raise ValueError(msg)

            image_grid.add_image(image_result.image, "Image")
            if image_result.gt_boxes is not None:
                gt_image = draw_boxes(np.copy(image_result.image), image_result.gt_boxes, color=(255, 0, 0))
                image_grid.add_image(image=gt_image, color_map="gray", title="Ground Truth")
            else:
                image_grid.add_image(image_result.image, "Image")
            pred_image = draw_boxes(np.copy(image_result.image), image_result.normal_boxes, color=(0, 255, 0))
            pred_image = draw_boxes(pred_image, image_result.anomalous_boxes, color=(255, 0, 0))
            image_grid.add_image(pred_image, "Predictions")
        if self.task == TaskType.SEGMENTATION:
            if image_result.pred_mask is None:
                msg = "Image result predicted mask is None."
                raise ValueError(msg)

            image_grid.add_image(image_result.image, "Image")
            if image_result.gt_mask is not None:
                image_grid.add_image(image=image_result.gt_mask, color_map="gray", title="Ground Truth")
            image_grid.add_image(image_result.heat_map, "Predicted Heat Map")
            image_grid.add_image(image=image_result.pred_mask, color_map="gray", title="Predicted Mask")
            image_grid.add_image(image=image_result.segmentations, title="Segmentation Result")
        elif self.task == TaskType.CLASSIFICATION:
            image_grid.add_image(image_result.image, title="Image")
            if image_result.heat_map is not None:
                image_grid.add_image(image_result.heat_map, "Predicted Heat Map")
            if image_result.pred_label:
                image_classified = add_anomalous_label(image_result.image, image_result.pred_score)
            else:
                image_classified = add_normal_label(image_result.image, 1 - image_result.pred_score)
            image_grid.add_image(image=image_classified, title="Prediction")

        return image_grid.generate()

    def _visualize_simple(self, image_result: ImageResult) -> np.ndarray:
        """Generate a simple visualization for an image.

        The simple visualization mode only shows the model's predictions in a single image.

        Args:
            image_result (ImageResult): GT and Prediction data for a single image.

        Returns:
            An image showing the simple visualization for the input image.
        """
        if self.task == TaskType.DETECTION:
            # return image with bounding boxes augmented
            image_with_boxes = draw_boxes(
                image=np.copy(image_result.image),
                boxes=image_result.anomalous_boxes,
                color=(0, 0, 255),
            )
            if image_result.gt_boxes is not None:
                image_with_boxes = draw_boxes(image=image_with_boxes, boxes=image_result.gt_boxes, color=(255, 0, 0))
            return image_with_boxes
        if self.task == TaskType.SEGMENTATION:
            visualization = mark_boundaries(
                image_result.heat_map,
                image_result.pred_mask,
                color=(1, 0, 0),
                mode="thick",
            )
            return (visualization * 255).astype(np.uint8)
        if self.task == TaskType.CLASSIFICATION:
            if image_result.pred_label:
                image_classified = add_anomalous_label(image_result.image, image_result.pred_score)
            else:
                image_classified = add_normal_label(image_result.image, 1 - image_result.pred_score)
            return image_classified
        msg = f"Unknown task type: {self.task}"
        raise ValueError(msg)


class _ImageGrid:
    """Helper class that compiles multiple images into a grid using subplots.

    Individual images can be added with the `add_image` method. When all images have been added, the `generate` method
    must be called to compile the image grid and obtain the final visualization.
    """

    def __init__(self) -> None:
        self.images: list[dict] = []
        self.figure: matplotlib.figure.Figure | None = None
        self.axis: Axes | np.ndarray | None = None

    def add_image(self, image: np.ndarray, title: str | None = None, color_map: str | None = None) -> None:
        """Add an image to the grid.

        Args:
          image (np.ndarray): Image which should be added to the figure.
          title (str): Image title shown on the plot.
          color_map (str | None): Name of matplotlib color map used to map scalar data to colours. Defaults to None.
        """
        image_data = {"image": image, "title": title, "color_map": color_map}
        self.images.append(image_data)

    def generate(self) -> np.ndarray:
        """Generate the image.

        Returns:
            Image consisting of a grid of added images and their title.
        """
        num_cols = len(self.images)
        figure_size = (num_cols * 5, 5)

        # Use Agg backend. This method fails when using backend like MacOSX which might be automatically selected
        # The dimension of image returned by tostring_rgb() does not match the dimension of the canvas
        matplotlib.use("Agg")

        self.figure, self.axis = plt.subplots(1, num_cols, figsize=figure_size)
        self.figure.subplots_adjust(right=0.9)

        axes = self.axis if isinstance(self.axis, np.ndarray) else np.array([self.axis])
        for axis, image_dict in zip(axes, self.images, strict=True):
            axis.axes.xaxis.set_visible(b=False)
            axis.axes.yaxis.set_visible(b=False)
            axis.imshow(image_dict["image"], image_dict["color_map"], vmin=0, vmax=255)
            if image_dict["title"] is not None:
                axis.title.set_text(image_dict["title"])
        self.figure.canvas.draw()
        # convert canvas to numpy array to prepare for visualization with opencv
        img = np.frombuffer(self.figure.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.figure.canvas.get_width_height()[::-1] + (3,))
        plt.close(self.figure)
        return img
