"""Image Visualizer."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from lightning.pytorch import Callback, Trainer

from anomalib.data import ImageBatch
from anomalib.models import AnomalyModule
from anomalib.utils.path import generate_output_filename

from .functional import visualize_image_item


class ImageVisualizer(Callback):
    """Image Visualizer.

    This class is responsible for visualizing images and their corresponding anomaly maps
    during the testing and prediction phases of an anomaly detection model.

    Args:
        fields (list[str] | None): List of fields to visualize.
            Defaults to ``["image", "gt_mask"]``.
        overlay_fields (list[tuple[str, list[str]]] | None): List of tuples specifying fields to overlay.
            Defaults to ``[("image", ["anomaly_map"]), ("image", ["pred_mask"])]``.
        field_size (tuple[int, int]): Size of each field in the visualization.
            Defaults to ``(256, 256)``.
        alpha (float): Alpha value for overlay blending.
            Defaults to ``0.2``.
        colormap (bool): Whether to apply a colormap to the anomaly maps.
            Defaults to ``True``.
        normalize (bool): Whether to normalize the anomaly maps.
            Defaults to ``False``.
        output_dir (str | Path | None): Directory to save the visualizations.
            Defaults to ``None``.

    Examples:
        >>> from anomalib.visualization import ImageVisualizer
        >>> from anomalib.data import MVTec
        >>> from anomalib.models import Patchcore
        >>> from anomalib.engine import Engine

        >>> # Create a basic visualizer
        >>> visualizer = ImageVisualizer()

        >>> # Create a visualizer with custom settings and fields
        >>> visualizer = ImageVisualizer(
        ...     fields=["image", "gt_mask", "anomaly_map", "pred_mask"],  # Customize fields
        ...     overlay_fields=[("image", ["anomaly_map"]), ("image", ["pred_mask"])],  # Customize overlays
        ...     field_size=(512, 512),
        ...     alpha=0.5,
        ...     colormap=True,
        ...     output_dir="./output/visualizations"
        ... )

        >>> # Set up your model, data and engine
        >>> model = Patchcore()
        >>> datamodule = MVTec()
        >>> engine = Engine(callbacks=[visualizer])

        >>> # Fit and test the model
        >>> trainer.fit(model, datamodule)
        >>> trainer.test(model, datamodule)

        After testing, you'll find the visualizations in the specified output directory.
        Each image will contain the fields specified in 'fields' and 'overlay_fields'.

        You can also use the visualizer during prediction:
        trainer.predict(model, datamodule)

        This will generate similar visualizations for the prediction data.

    Note:
        - The visualizer automatically handles both test and predict scenarios.
        - It saves the visualizations to the specified output directory or to a default
          location if not specified.
        - The 'fields' and 'overlay_fields' arguments allow for customization of the
          visualization layout. You can include or exclude specific fields based on
          your requirements.
    """

    def __init__(
        self,
        fields: list[str] | None = None,
        *,
        overlay_fields: list[tuple[str, list[str]]] | None = None,
        field_size: tuple[int, int] = (256, 256),
        alpha: float = 0.2,
        colormap: bool = True,
        normalize: bool = False,
        output_dir: str | Path | None = None,
    ) -> None:
        self.fields = fields or ["image", "gt_mask"]
        self.overlay_fields = overlay_fields or [("image", ["anomaly_map"]), ("image", ["pred_mask"])]
        self.field_size = field_size
        self.alpha = alpha
        self.colormap = colormap
        self.normalize = normalize
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
                alpha=self.alpha,
                colormap=self.colormap,
                normalize=self.normalize,
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
