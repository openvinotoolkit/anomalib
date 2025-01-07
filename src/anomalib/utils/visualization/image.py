"""Image and video visualization generator.

This module provides utilities for visualizing anomaly detection results on images
and videos. The key components include:

    - ``ImageResult``: Dataclass for storing visualization data
    - ``ImageVisualizer``: Main visualization generator class
    - ``VisualizationMode``: Enum for controlling visualization style
    - ``_ImageGrid``: Helper class for creating image grids

The module supports both classification and segmentation tasks, with options for:

    - Full visualization showing all available outputs
    - Simple visualization showing only key predictions
    - Customizable normalization of anomaly maps
    - Automatic handling of both image and video inputs

Example:
    >>> from anomalib.utils.visualization import ImageVisualizer
    >>> from anomalib.utils.visualization.image import VisualizationMode
    >>> # Create visualizer
    >>> visualizer = ImageVisualizer(
    ...     mode=VisualizationMode.FULL,
    ...     task="segmentation",
    ...     normalize=True
    ... )
    >>> # Generate visualization
    >>> results = visualizer.generate(
    ...     outputs={
    ...         "image": images,
    ...         "pred_mask": masks,
    ...         "anomaly_map": heatmaps
    ...     }
    ... )

The module ensures consistent visualization across different anomaly detection
approaches and result types. It handles proper scaling and formatting of inputs,
and provides a flexible interface for customizing the visualization output.

Note:
    When using video inputs, the visualizer automatically handles frame extraction
    and maintains proper frame ordering in the output.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Iterator
from dataclasses import InitVar, asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries

from anomalib import TaskType
from anomalib.data import ImageItem, NumpyImageItem, VideoItem
from anomalib.data.utils import read_image
from anomalib.utils.post_processing import add_anomalous_label, add_normal_label, superimpose_anomaly_map

from .base import BaseVisualizer, GeneratorResult, VisualizationStep

if TYPE_CHECKING:
    from matplotlib.axis import Axes


class VisualizationMode(str, Enum):
    """Visualization mode for controlling output style.

    The mode determines how results are displayed:

    - ``FULL``: Shows all available visualizations in a grid
    - ``SIMPLE``: Shows only the key prediction results
    """

    FULL = "full"
    SIMPLE = "simple"


@dataclass
class ImageResult:
    """Collection of data needed to visualize predictions for an image.

    Args:
        image (np.ndarray): Input image to visualize
        pred_score (float): Predicted anomaly score
        pred_label (str): Predicted label (e.g. "normal" or "anomalous")
        anomaly_map (np.ndarray | None): Anomaly heatmap if available
        gt_mask (np.ndarray | None): Ground truth mask if available
        pred_mask (np.ndarray | None): Predicted segmentation mask if available
        normalize (InitVar[bool]): Whether to normalize anomaly maps to [0,1]

    Note:
        The class automatically handles proper scaling and type conversion of
        inputs during initialization.
    """

    image: np.ndarray
    pred_score: float
    pred_label: str
    anomaly_map: np.ndarray | None = None
    gt_mask: np.ndarray | None = None
    pred_mask: np.ndarray | None = None
    normalize: InitVar[bool] = False

    def __post_init__(self, normalize: bool) -> None:
        """Format and compute additional fields."""
        if self.image.dtype != np.uint8:
            self.image = (self.image * 255).astype(np.uint8)
        if self.anomaly_map is not None:
            height, width = self.anomaly_map.squeeze().shape[:2]
            self.image = cv2.resize(self.image.squeeze(), (width, height))

        if self.anomaly_map is not None:
            self.heat_map = superimpose_anomaly_map(self.anomaly_map, self.image, normalize=normalize)
        else:
            self.heat_map = None

        if self.gt_mask is not None and self.gt_mask.max() <= 1.0:
            if self.gt_mask.dtype == bool:
                self.gt_mask = self.gt_mask.astype(np.uint8)
            self.gt_mask *= 255

        if self.pred_mask is not None:
            self.pred_mask = self.pred_mask.astype(np.uint8).squeeze()
            if self.pred_mask.max() <= 1.0:
                self.pred_mask *= 255
                self.segmentations = mark_boundaries(self.image, self.pred_mask, color=(1, 0, 0), mode="thick")
                if self.segmentations.max() <= 1.0:
                    self.segmentations = (self.segmentations * 255).astype(np.uint8)

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        repr_str = (
            f"ImageResult(image={self.image}, pred_score={self.pred_score}, pred_label={self.pred_label}, "
            f"anomaly_map={self.anomaly_map}, gt_mask={self.gt_mask}, "
        )
        repr_str += f", pred_mask={self.pred_mask}" if self.pred_mask is not None else ""
        repr_str += f", heat_map={self.heat_map}" if self.heat_map is not None else ""
        repr_str += f", segmentations={self.segmentations}" if self.segmentations is not None else ""
        repr_str += ")"
        return repr_str

    @classmethod
    def from_dataset_item(cls: type["ImageResult"], item: ImageItem | NumpyImageItem) -> "ImageResult":
        """Create an ImageResult object from a DatasetItem object.

        This is a temporary solution until we refactor the visualizer to take a
        DatasetItem object directly as input.

        Args:
            item (ImageItem | NumpyImageItem): Dataset item to convert

        Returns:
            ImageResult: New image result object
        """
        if isinstance(item, ImageItem):
            item = item.to_numpy()
        item_dict = asdict(item)
        field_names = {field.name for field in fields(cls)} & set(item_dict.keys())
        return cls(**{key: item_dict[key] for key in field_names})


class ImageVisualizer(BaseVisualizer):
    """Image and video visualization generator.

    Args:
        mode (VisualizationMode, optional): Visualization mode. Defaults to
            ``VisualizationMode.FULL``.
        task (TaskType | str, optional): Type of task. Defaults to
            ``TaskType.CLASSIFICATION``.
        normalize (bool, optional): Whether to normalize anomaly maps to image
            min-max. Defaults to ``False``.

    Note:
        Normalization is most useful when no other normalization method is used,
        as otherwise the overlay will show raw anomaly scores.
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
        """Generate images and return them as an iterator.

        Args:
            **kwargs: Keyword arguments containing model outputs.

        Returns:
            Iterator yielding visualization results.

        Raises:
            ValueError: If outputs are not provided in kwargs.
        """
        outputs = kwargs.get("outputs", None)
        if outputs is None:
            msg = "Outputs must be provided to generate images."
            raise ValueError(msg)
        return self._visualize_batch(outputs)

    def _visualize_batch(self, batch: dict) -> Iterator[GeneratorResult]:
        """Yield a visualization result for each item in the batch.

        Args:
            batch (dict): Dictionary containing the ground truth and predictions
                of a batch of images.

        Returns:
            Generator that yields a display-ready visualization for each image.

        Raises:
            TypeError: If item has neither image path nor video path defined.
        """
        for item in batch:
            if hasattr(item, "image_path") and item.image_path is not None:
                image = read_image(path=item.image_path, as_tensor=True)
                # set filename
                file_name = Path(item.image_path)
            elif hasattr(item, "video_path") and item.video_path is not None:
                image = item.original_image
                # set filename
                zero_fill = int(np.log10(item.last_frame.cpu())) + 1
                suffix = f"{str(item.frames.int().item()).zfill(zero_fill)}.png"
                file_name = Path(item.video_path) / suffix
            else:
                msg = "Item must have image path or video path defined."
                raise TypeError(msg)

            item.image = image
            if isinstance(item, VideoItem):
                image_result = ImageResult.from_dataset_item(item.to_image())
            else:
                image_result = ImageResult.from_dataset_item(item)
            yield GeneratorResult(image=self.visualize_image(image_result), file_name=file_name)

    def visualize_image(self, image_result: ImageResult) -> np.ndarray:
        """Generate the visualization for an image.

        Args:
            image_result (ImageResult): GT and Prediction data for a single image.

        Returns:
            np.ndarray: The full or simple visualization for the image.

        Raises:
            ValueError: If visualization mode is unknown.
        """
        if self.mode == VisualizationMode.FULL:
            return self._visualize_full(image_result)
        if self.mode == VisualizationMode.SIMPLE:
            return self._visualize_simple(image_result)
        msg = f"Unknown visualization mode: {self.mode}"
        raise ValueError(msg)

    def _visualize_full(self, image_result: ImageResult) -> np.ndarray:
        """Generate the full set of visualization for an image.

        The full visualization mode shows a grid with subplots that contain:
            - Original image
            - GT mask (if available)
            - Predicted heat map
            - Predicted segmentation mask (if available)
            - Predicted segmentations (if available)

        Args:
            image_result (ImageResult): GT and Prediction data for a single image.

        Returns:
            np.ndarray: Image showing the full set of visualizations.

        Raises:
            ValueError: If predicted mask is None for segmentation task.
        """
        image_grid = _ImageGrid()
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

        The simple visualization mode only shows the model's predictions in a
        single image.

        Args:
            image_result (ImageResult): GT and Prediction data for a single image.

        Returns:
            np.ndarray: Image showing the simple visualization.

        Raises:
            ValueError: If task type is unknown.
        """
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

    Individual images can be added with the ``add_image`` method. When all images
    have been added, the ``generate`` method must be called to compile the image
    grid and obtain the final visualization.
    """

    def __init__(self) -> None:
        self.images: list[dict] = []
        self.figure: matplotlib.figure.Figure | None = None
        self.axis: Axes | np.ndarray | None = None

    def add_image(self, image: np.ndarray, title: str | None = None, color_map: str | None = None) -> None:
        """Add an image to the grid.

        Args:
            image (np.ndarray): Image to add to the figure
            title (str | None): Image title shown on the plot
            color_map (str | None): Name of matplotlib color map for mapping
                scalar data to colours. Defaults to ``None``.
        """
        image_data = {"image": image, "title": title, "color_map": color_map}
        self.images.append(image_data)

    def generate(self) -> np.ndarray:
        """Generate the image grid.

        Returns:
            np.ndarray: Image consisting of a grid of added images and their
            titles.

        Note:
            Uses Agg backend to avoid issues with dimension mismatch when using
            backends like MacOSX.
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
        img = np.array(self.figure.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
        plt.close(self.figure)
        return img
