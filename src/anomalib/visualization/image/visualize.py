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
    """Image Visualizer."""

    def __init__(
        self,
        fields: list[str] | None = None,
        *,
        overlay_fields: list[tuple[str, list[str]]] | None = None,
        field_size: tuple[int, int] = (256, 256),
        alpha: float = 0.5,
        colormap: bool = True,
        normalize: bool = True,
        output_dir: str | Path | None = None,
    ) -> None:
        self.fields = fields or ["image", "anomaly_map", "gt_mask", "pred_mask"]
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
        del pl_module, batch, batch_idx, dataloader_idx  # Unused arguments.

        if not isinstance(outputs, ImageBatch):
            msg = "Outputs must be an instance of ImageBatch"
            raise TypeError(msg)

        if self.output_dir is None:
            self.output_dir = Path(trainer.default_root_dir) / "images"

        for output in outputs:
            image = visualize_image_item(
                output,
                fields=self.fields,
                overlay_fields=self.overlay_fields,
                alpha=self.alpha,
                colormap=self.colormap,
                normalize=self.normalize,
            )
            if image is not None:
                # Get the dataset name and category to save the image
                filename = generate_output_filename(
                    input_path=output.image_path or "",
                    output_path=self.output_dir,
                    dataset_name=trainer.test_dataloaders.dataset.name or "",
                    category=trainer.test_dataloaders.dataset.category or "",
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
