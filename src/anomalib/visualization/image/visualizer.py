"""Image Visualizer."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

from lightning.pytorch import Callback, Trainer

from anomalib.data import ImageBatch
from anomalib.models import AnomalyModule
from anomalib.utils.path import generate_output_filename

from .item_visualizer import (
    DEFAULT_FIELDS_CONFIG,
    DEFAULT_OVERLAY_FIELDS_CONFIG,
    DEFAULT_TEXT_CONFIG,
    visualize_image_item,
)


class ImageVisualizer(Callback):
    """Image Visualizer.

    This class is responsible for visualizing images and their corresponding anomaly maps
    during the testing and prediction phases of an anomaly detection model.

    Args:
        fields (list[str] | None): List of fields to visualize.
            Defaults to ["image", "gt_mask"].
        overlay_fields (list[tuple[str, list[str]]] | None): List of tuples specifying fields to overlay.
            Defaults to [("image", ["anomaly_map"]), ("image", ["pred_mask"])].
        field_size (tuple[int, int]): Size of each field in the visualization.
            Defaults to (256, 256).
        fields_config (dict[str, dict[str, Any]]): Custom configurations for field visualization.
            Defaults to DEFAULT_FIELDS_CONFIG.
        overlay_fields_config (dict[str, dict[str, Any]]): Custom configurations for field overlays.
            Defaults to DEFAULT_OVERLAY_FIELDS_CONFIG.
        text_config (dict[str, Any]): Configuration for text overlay.
            Defaults to DEFAULT_TEXT_CONFIG.
        output_dir (str | Path | None): Directory to save the visualizations.
            Defaults to None.

    Examples:
        Basic usage with default settings:
        >>> visualizer = ImageVisualizer()

        Customizing fields to visualize:
        >>> visualizer = ImageVisualizer(
        ...     fields=["image", "gt_mask", "anomaly_map"],
        ...     overlay_fields=[("image", ["anomaly_map"])]
        ... )

        Adjusting field size:
        >>> visualizer = ImageVisualizer(field_size=(512, 512))

        Customizing anomaly map visualization:
        >>> visualizer = ImageVisualizer(
        ...     fields_config={
        ...         "anomaly_map": {"colormap": True, "normalize": True}
        ...     }
        ... )

        Modifying overlay appearance:
        >>> visualizer = ImageVisualizer(
        ...     overlay_fields_config={
        ...         "pred_mask": {"alpha": 0.7, "color": (255, 0, 0), "mode": "fill"},
        ...         "anomaly_map": {"alpha": 0.5, "color": (0, 255, 0), "mode": "contour"}
        ...     }
        ... )

        Customizing text overlay:
        >>> visualizer = ImageVisualizer(
        ...     text_config={
        ...         "font": "arial.ttf",
        ...         "size": 20,
        ...         "color": "yellow",
        ...         "background": (0, 0, 0, 200)
        ...     }
        ... )

        Specifying output directory:
        >>> visualizer = ImageVisualizer(output_dir="./output/visualizations")

        Advanced configuration combining multiple customizations:
        >>> visualizer = ImageVisualizer(
        ...     fields=["image", "gt_mask", "anomaly_map", "pred_mask"],
        ...     overlay_fields=[("image", ["anomaly_map"]), ("image", ["pred_mask"])],
        ...     field_size=(384, 384),
        ...     fields_config={
        ...         "anomaly_map": {"colormap": True, "normalize": True},
        ...         "pred_mask": {"color": (0, 0, 255)}
        ...     },
        ...     overlay_fields_config={
        ...         "anomaly_map": {"alpha": 0.6, "mode": "fill"},
        ...         "pred_mask": {"alpha": 0.7, "mode": "contour"}
        ...     },
        ...     text_config={
        ...         "font": "times.ttf",
        ...         "size": 24,
        ...         "color": "white",
        ...         "background": (0, 0, 0, 180)
        ...     },
        ...     output_dir="./custom_visualizations"
        ... )

    Note:
        - The 'fields' parameter determines which individual fields are visualized.
        - The 'overlay_fields' parameter specifies which fields should be overlaid on others.
        - Field configurations in 'fields_config' affect how individual fields are visualized.
        - Overlay configurations in 'overlay_fields_config' determine how fields are blended when overlaid.
        - Text configurations in 'text_config' control the appearance of text labels on visualizations.
        - If 'output_dir' is not specified, visualizations will be saved in a default location.

    For more details on available options for each configuration, refer to the documentation
    of the `visualize_image_item`, `visualize_field`, and related functions.
    """

    def __init__(
        self,
        fields: list[str] | None = None,
        overlay_fields: list[tuple[str, list[str]]] | None = None,
        field_size: tuple[int, int] = (256, 256),
        fields_config: dict[str, dict[str, Any]] | None = None,
        overlay_fields_config: dict[str, dict[str, Any]] | None = None,
        text_config: dict[str, Any] | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.fields = fields or ["image", "gt_mask"]
        self.overlay_fields = overlay_fields or [("image", ["anomaly_map"]), ("image", ["pred_mask"])]
        self.field_size = field_size
        self.fields_config = {**DEFAULT_FIELDS_CONFIG, **(fields_config or {})}
        self.overlay_fields_config = {**DEFAULT_OVERLAY_FIELDS_CONFIG, **(overlay_fields_config or {})}
        self.text_config = {**DEFAULT_TEXT_CONFIG, **(text_config or {})}
        self.output_dir = output_dir

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: AnomalyModule,
        outputs: ImageBatch,
        batch: ImageBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the test batch ends."""
        del pl_module, outputs, batch_idx, dataloader_idx  # Unused arguments.

        if self.output_dir is None:
            self.output_dir = Path(trainer.default_root_dir) / "images"

        for item in batch:
            image = visualize_image_item(
                item,
                fields=self.fields,
                overlay_fields=self.overlay_fields,
                field_size=self.field_size,
                fields_config=self.fields_config,
                overlay_fields_config=self.overlay_fields_config,
                text_config=self.text_config,
            )

            if image is not None:
                # Get the dataset name and category to save the image
                filename = generate_output_filename(
                    input_path=item.image_path or "",
                    output_path=self.output_dir,
                    dataset_name=getattr(trainer.test_dataloaders.dataset, "name", "") or "",
                    category=getattr(trainer.test_dataloaders.dataset, "category", "") or "",
                )

                # Save the image to the specified filename
                image.save(filename)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: AnomalyModule,
        outputs: ImageBatch,
        batch: ImageBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the predict batch ends."""
        return self.on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
