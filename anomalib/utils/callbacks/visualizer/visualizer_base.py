"""Base Visualizer Callback."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Union, cast

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Callback

from anomalib.models.components import AnomalyModule
from anomalib.post_processing import Visualizer
from anomalib.utils.loggers import AnomalibWandbLogger
from anomalib.utils.loggers.base import ImageLoggerBase


class BaseVisualizerCallback(Callback):
    """Callback that visualizes the results of a model.

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
        filename: Union[Path, str],
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
                    if isinstance(filename, Path):
                        _name = filename.parent.name + "_" + filename.name
                    elif isinstance(filename, str):
                        _name = filename
                    logger.add_image(
                        image=image,
                        name=_name,
                        global_step=module.global_step,
                    )

    def on_test_end(self, trainer: pl.Trainer, pl_module: AnomalyModule) -> None:
        """Sync logs.

        Currently only ``AnomalibWandbLogger.save`` is called from this method.
        This is because logging as a single batch ensures that all images appear as part of the same step.

        Args:
            trainer (pl.Trainer): Pytorch Lightning trainer
            pl_module (AnomalyModule): Anomaly module (unused)
        """
        for logger in trainer.loggers:
            if isinstance(logger, AnomalibWandbLogger):
                logger.save()
