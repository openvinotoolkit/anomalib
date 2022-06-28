"""Visualizer Callback."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY
from pytorch_lightning.utilities.types import STEP_OUTPUT

from anomalib.models.components import AnomalyModule
from anomalib.post_processing import Visualizer
from anomalib.utils.loggers import AnomalibWandbLogger
from anomalib.utils.loggers.base import ImageLoggerBase


@CALLBACK_REGISTRY
class VisualizerCallback(Callback):
    """Callback that visualizes the inference results of a model.

    The callback generates a figure showing the original image, the ground truth segmentation mask,
    the predicted error heat map, and the predicted segmentation mask.

    To save the images to the filesystem, add the 'local' keyword to the `project.log_images_to` parameter in the
    config.yaml file.
    """

    def __init__(
        self,
        task: str,
        mode: str,
        image_save_path: str,
        inputs_are_normalized: bool = True,
        show_images: bool = False,
        log_images: bool = True,
        save_images: bool = True,
    ):
        """Visualizer callback."""
        if mode not in ["full", "simple"]:
            raise ValueError(f"Unknown visualization mode: {mode}. Please choose one of ['full', 'simple']")
        self.mode = mode
        if task not in ["classification", "segmentation"]:
            raise ValueError(f"Unknown task type: {mode}. Please choose one of ['classification', 'segmentation']")
        self.task = task
        self.inputs_are_normalized = inputs_are_normalized
        self.show_images = show_images
        self.log_images = log_images
        self.save_images = save_images
        self.image_save_path = Path(image_save_path)

        self.visualizer = Visualizer(mode, task)

    def _add_to_logger(
        self,
        image: np.ndarray,
        module: AnomalyModule,
        trainer: pl.Trainer,
        filename: Path,
    ):
        """Log image from a visualizer to each of the available loggers in the project.

        Args:
            image (np.ndarray): Image that should be added to the loggers.
            module (AnomalyModule): Anomaly module.
            trainer (Trainer): Pytorch Lightning trainer which holds reference to `logger`
            filename (Path): Path of the input image. This name is used as name for the generated image.
        """
        # Store names of logger and the logger in a dict
        available_loggers = {
            type(logger).__name__.lower().rstrip("logger").lstrip("anomalib"): logger for logger in trainer.loggers
        }
        # save image to respective logger
        if self.log_images:
            for log_to in available_loggers:
                # check if logger object is same as the requested object
                if isinstance(available_loggers[log_to], ImageLoggerBase):
                    logger: ImageLoggerBase = cast(ImageLoggerBase, available_loggers[log_to])  # placate mypy
                    logger.add_image(
                        image=image,
                        name=filename.parent.name + "_" + filename.name,
                        global_step=module.global_step,
                    )

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

    def on_test_end(self, _trainer: pl.Trainer, pl_module: AnomalyModule) -> None:
        """Sync logs.

        Currently only ``AnomalibWandbLogger`` is called from this method. This is because logging as a single batch
        ensures that all images appear as part of the same step.

        Args:
            _trainer (pl.Trainer): Pytorch Lightning trainer (unused)
            pl_module (AnomalyModule): Anomaly module
        """
        if pl_module.logger is not None and isinstance(pl_module.logger, AnomalibWandbLogger):
            pl_module.logger.save()
