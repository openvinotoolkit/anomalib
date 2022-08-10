"""Image Visualizer Callback."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY
from pytorch_lightning.utilities.types import STEP_OUTPUT

from anomalib.models.components import AnomalyModule

from .visualizer_base import BaseVisualizerCallback


@CALLBACK_REGISTRY
class ImageVisualizerCallback(BaseVisualizerCallback):
    """Callback that visualizes the inference results of a model.

    The callback generates a figure showing the original image, the ground truth segmentation mask,
    the predicted error heat map, and the predicted segmentation mask.

    To save the images to the filesystem, add the 'local' keyword to the `project.log_images_to` parameter in the
    config.yaml file.
    """

    def on_predict_batch_end(
        self,
        _trainer: pl.Trainer,
        _pl_module: AnomalyModule,
        outputs: Optional[STEP_OUTPUT],
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Show images at the end of every batch.

        Args:
            _trainer (Trainer): Pytorch lightning trainer object (unused).
            _pl_module (LightningModule): Lightning modules derived from BaseAnomalyLightning object as
            currently only they support logging images.
            outputs (Dict[str, Any]): Outputs of the current test step.
            _batch (Any): Input batch of the current test step (unused).
            _batch_idx (int): Index of the current test batch (unused).
            _dataloader_idx (int): Index of the dataloader that yielded the current batch (unused).
        """
        assert outputs is not None
        for i, image in enumerate(self.visualizer.visualize_batch(outputs)):
            filename = Path(outputs["image_path"][i])
            if self.save_images:
                file_path = self.image_save_path / filename.parent.name / filename.name
                self.visualizer.save(file_path, image)
            if self.show_images:
                self.visualizer.show(str(filename), image)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: AnomalyModule,
        outputs: Optional[STEP_OUTPUT],
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Log images at the end of every batch.

        Args:
            trainer (Trainer): Pytorch lightning trainer object (unused).
            pl_module (LightningModule): Lightning modules derived from BaseAnomalyLightning object as
            currently only they support logging images.
            outputs (Dict[str, Any]): Outputs of the current test step.
            _batch (Any): Input batch of the current test step (unused).
            _batch_idx (int): Index of the current test batch (unused).
            _dataloader_idx (int): Index of the dataloader that yielded the current batch (unused).
        """
        assert outputs is not None
        for i, image in enumerate(self.visualizer.visualize_batch(outputs)):
            filename = Path(outputs["image_path"][i])
            if self.save_images:
                file_path = self.image_save_path / filename.parent.name / filename.name
                self.visualizer.save(file_path, image)
            if self.log_images:
                self._add_to_logger(image, pl_module, trainer, filename)
            if self.show_images:
                self.visualizer.show(str(filename), image)
